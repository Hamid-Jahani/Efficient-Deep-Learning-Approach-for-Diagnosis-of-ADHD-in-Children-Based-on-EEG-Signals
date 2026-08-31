"""Train/test splitting and k-fold cross-validation, extracted from
``ADHD.ipynb`` (cells 23-37).

Epochs are grouped into fixed-size *segments*, and each subject contributes
several segments. Two splitting strategies are available:

``split_by="segment"``
    The default, reproducing the published pipeline. Segments are assigned to
    folds independently of which subject they came from.

``split_by="subject"``
    Segments are grouped by subject, so all of a subject's segments stay on the
    same side of a split.

Predictions can likewise be aggregated per segment or per subject via
``aggregate_by``. Every result records the pair actually used.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils import shuffle

from adhd_eeg.resnets_utils import convert_to_one_hot

__all__ = [
    "SplitStrategy",
    "segment_table",
    "build_folds",
    "subjects_in",
    "cross_validate",
]

SplitStrategy = Literal["segment", "subject"]


def _segment_membership(full_data: pd.DataFrame, segment_size: int):
    """Split segments into "complete" (>= segment_size epochs) and "partial".

    Returns ``(partial_segment_ids, complete_segment_frame)``.

    ``partials`` is returned as a plain list. The previous implementation built
    it with ``.index.to_series(name="segment")``, producing a *Series*, and then
    did ``list(partials.segment)`` - a Series has no ``.segment`` attribute, so
    that raised ``AttributeError`` on the first fold. Nothing caught it because
    running ``cross_validate`` needs TensorFlow.

    ``dummy_y`` is a constant 0 column used only to satisfy ``StratifiedKFold``'s
    API. Because it is constant the split is *not* stratified by class - it
    degenerates to plain ``KFold``. Preserved for the ``segment`` strategy so
    published results stay reproducible.
    """
    counts = full_data.groupby("segment")["Y"].count()
    partials = counts.index[counts < segment_size].tolist()
    complete = counts.index[counts >= segment_size].to_series().reset_index(drop=True)
    complete_segment = pd.DataFrame({"segment": complete, "dummy_y": 0})
    return partials, complete_segment


def segment_table(full_data: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: its subject and its (single) class label.

    Raises ``ValueError`` if a segment spans more than one subject or carries
    more than one label, which would make grouping meaningless.
    """
    grouped = full_data.groupby("segment").agg(
        person=("person", "nunique"),
        label=("Y", "nunique"),
    )
    bad_subject = grouped.index[grouped["person"] > 1].tolist()
    if bad_subject:
        raise ValueError(f"segment(s) {bad_subject[:5]} span more than one subject")
    bad_label = grouped.index[grouped["label"] > 1].tolist()
    if bad_label:
        raise ValueError(f"segment(s) {bad_label[:5]} carry more than one label")

    return (
        full_data.groupby("segment")
        .agg(person=("person", "first"), Y=("Y", "first"))
        .reset_index()
    )


def subjects_in(full_data: pd.DataFrame, segment_ids: Sequence) -> set:
    """Return the set of subject ids covered by ``segment_ids``."""
    return set(full_data.loc[full_data.segment.isin(list(segment_ids)), "person"])


def build_folds(
    full_data: pd.DataFrame,
    *,
    segment_size: int = 20,
    n_splits: int = 10,
    random_state: int = 10,
    split_by: SplitStrategy = "segment",
) -> List[Tuple[List, List]]:
    """Return ``[(train_segments, test_segments), ...]`` for each fold.

    Undersized "partial" segments always go to training, matching the notebook.

    ``split_by="segment"``
        The published behaviour. Segments are assigned to folds independently
        of which subject they came from.

    ``split_by="subject"``
        Segments are split with ``StratifiedGroupKFold`` grouped by subject, so
        every segment of a subject stays on one side, while class balance is
        maintained across folds.
    """
    partials, complete_segment = _segment_membership(full_data, segment_size)
    partial_list = list(partials)

    if split_by == "segment":
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kfold.split(complete_segment.segment, complete_segment.dummy_y)
        folds = []
        for train_idx, test_idx in splits:
            folds.append(
                (
                    list(complete_segment.segment.loc[train_idx]) + partial_list,
                    list(complete_segment.segment.loc[test_idx]),
                )
            )
        return folds

    if split_by != "subject":
        raise ValueError(f"unknown split_by {split_by!r}; expected 'segment' or 'subject'")

    table = segment_table(full_data)
    table = table[table.segment.isin(set(complete_segment.segment))].reset_index(drop=True)

    n_subjects = table.person.nunique()
    if n_subjects < n_splits:
        raise ValueError(
            f"cannot make {n_splits} subject-wise folds from {n_subjects} subject(s); "
            f"pass n_splits <= {n_subjects}"
        )

    kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, test_idx in kfold.split(table.segment, table.Y, groups=table.person):
        folds.append(
            (
                list(table.segment.iloc[train_idx]) + partial_list,
                list(table.segment.iloc[test_idx]),
            )
        )
    return folds


def _split_xy(full_data: pd.DataFrame, images: np.ndarray, segment_ids) -> tuple:
    mask = full_data.segment.isin(list(segment_ids))
    y = np.array(full_data.Y[mask])
    idx = full_data.reset_index()["index"][mask]
    x = images[idx]
    return x, y


def _majority_vote(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate epoch predictions to one prediction per ``by`` group."""
    aggregated = (
        frame.groupby(by)
        .agg([lambda column: column.value_counts().index[0]])
        .reset_index()
    )
    aggregated.columns = [by, "Y", "y_pred"]
    return aggregated


def cross_validate(
    images: np.ndarray,
    labels: np.ndarray,
    subjects: List[int],
    segments: List[int],
    model_fn: Callable[..., object],
    input_shape=(19, 256, 3),
    segment_size: int = 20,
    n_splits: int = 10,
    epochs: int = 100,
    batch_size: int = 128,
    patience: int = 4,
    random_state: int = 10,
    verbose: int = 0,
    split_by: SplitStrategy = "segment",
    aggregate_by: Literal["segment", "person"] = "segment",
) -> List[Dict[str, object]]:
    """Run k-fold cross-validation and return one result dict per fold.

    ``split_by``
        ``"segment"`` (default) reproduces the published pipeline;
        ``"subject"`` keeps each subject's segments on one side of the split.

    ``aggregate_by``
        Level at which epoch predictions are majority-voted before scoring.
        ``"segment"`` (default) matches the published code; ``"person"``
        aggregates per child.

    Every result dict records the ``split_by`` and ``aggregate_by`` used, so a
    set of results cannot be mistaken for the other configuration.
    """
    import tensorflow as tf

    full_data = pd.DataFrame({"Y": labels, "person": subjects, "segment": segments})
    folds = build_folds(
        full_data,
        segment_size=segment_size,
        n_splits=n_splits,
        random_state=random_state,
        split_by=split_by,
    )

    results = []
    for train_segments, test_segments in folds:
        x_train, y_train = _split_xy(full_data, images, train_segments)
        x_test, y_test = _split_xy(full_data, images, test_segments)
        # Seeded: the original called shuffle() without a random_state, so runs
        # were not reproducible even with random_state fixed for the splitter.
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)

        model = model_fn(input_shape=input_shape, classes=2)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience)
        model.fit(
            x_train,
            convert_to_one_hot(y_train, 2).T,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[callback],
            verbose=verbose,
        )

        loss, epoch_accuracy = model.evaluate(
            x_test, convert_to_one_hot(y_test, 2).T, verbose=verbose
        )

        test_mask = full_data.segment.isin(list(test_segments))
        scored = full_data.loc[test_mask, [aggregate_by, "Y"]].copy()
        scored["y_pred"] = model.predict(x_test, verbose=verbose).argmax(axis=1)
        scored = _majority_vote(scored, aggregate_by)

        results.append(
            {
                "split_by": split_by,
                "aggregate_by": aggregate_by,
                "n_test_subjects": len(subjects_in(full_data, test_segments)),
                "epoch_loss": loss,
                "epoch_accuracy": epoch_accuracy,
                "accuracy": metrics.accuracy_score(scored.Y, scored.y_pred),
                "precision": metrics.precision_score(scored.Y, scored.y_pred),
                "recall": metrics.recall_score(scored.Y, scored.y_pred),
                "f1": metrics.f1_score(scored.Y, scored.y_pred),
                "confusion_matrix": metrics.confusion_matrix(scored.Y, scored.y_pred),
            }
        )

    return results
