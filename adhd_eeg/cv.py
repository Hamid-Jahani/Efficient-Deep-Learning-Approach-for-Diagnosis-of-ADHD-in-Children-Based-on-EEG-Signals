"""Segment-based train/test splitting and k-fold cross-validation, extracted
from ADHD.ipynb (cells 23-37).

Epochs are grouped into fixed-size "segments" per subject (see
``adhd_eeg.features.build_dataset``); splitting happens at the segment level
so that epochs from the same short time window never leak between train and
test. Any leftover, undersized segment ("partial") is always folded into the
training set, matching the notebook.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from adhd_eeg.resnets_utils import convert_to_one_hot


def _segment_membership(full_data: pd.DataFrame, segment_size: int):
    """Split segments into "complete" (>= segment_size epochs) and "partial".

    Note: matching the original notebook, ``dummy_y`` is a constant 0 column
    used only to satisfy ``StratifiedKFold``'s API; the fold split is not
    actually stratified by class label. This is preserved as-is for fidelity
    to the published pipeline rather than "fixed", since changing it would
    change the reported cross-validation results.
    """
    counts = full_data.groupby("segment")["Y"].count()
    partials = counts[counts < segment_size].index.to_series(name="segment").reset_index(drop=True)
    complete = counts[counts >= segment_size].index.to_series(name="segment").reset_index(drop=True)
    complete_segment = pd.DataFrame({"segment": complete, "dummy_y": 0})
    return partials, complete_segment


def _split_xy(full_data: pd.DataFrame, images: np.ndarray, segment_ids) -> tuple:
    mask = full_data.segment.isin(segment_ids)
    y = np.array(full_data.Y[mask])
    idx = full_data.reset_index()["index"][mask]
    x = images[idx]
    return x, y


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
) -> List[Dict[str, object]]:
    """Run stratified k-fold cross-validation at the segment level.

    ``model_fn`` should be ``adhd_eeg.models.cnn_model`` or
    ``adhd_eeg.models.resnet_model`` (or any zero/two-arg callable with the
    same ``(input_shape, classes)`` signature returning a compiled-able
    ``keras.Model``).

    Returns one result dict per fold with epoch-level accuracy/loss plus
    subject-level (majority-vote per segment) accuracy/precision/recall/f1
    and confusion matrix, mirroring the metrics reported in the notebook.
    """
    import tensorflow as tf

    full_data = pd.DataFrame({"Y": labels, "person": subjects, "segment": segments})
    partials, complete_segment = _segment_membership(full_data, segment_size)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    for train_idx, test_idx in kfold.split(complete_segment.segment, complete_segment.dummy_y):
        train_segments = list(complete_segment.segment.loc[train_idx]) + list(partials.segment)
        test_segments = list(complete_segment.segment.loc[test_idx])

        x_train, y_train = _split_xy(full_data, images, train_segments)
        x_test, y_test = _split_xy(full_data, images, test_segments)
        x_train, y_train = shuffle(x_train, y_train)

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

        loss, epoch_accuracy = model.evaluate(x_test, convert_to_one_hot(y_test, 2).T, verbose=verbose)

        segment_data = full_data[["segment", "Y"]][full_data.segment.isin(test_segments)].copy()
        segment_data["y_pred"] = model.predict(x_test, verbose=verbose).argmax(axis=1)
        segment_data = (
            segment_data.groupby("segment")
            .agg([lambda x: x.value_counts().index[0]])
            .reset_index()
        )
        segment_data.columns = ["segment", "Y", "y_pred"]

        results.append(
            {
                "epoch_loss": loss,
                "epoch_accuracy": epoch_accuracy,
                "accuracy": metrics.accuracy_score(segment_data.Y, segment_data.y_pred),
                "precision": metrics.precision_score(segment_data.Y, segment_data.y_pred),
                "recall": metrics.recall_score(segment_data.Y, segment_data.y_pred),
                "f1": metrics.f1_score(segment_data.Y, segment_data.y_pred),
                "confusion_matrix": metrics.confusion_matrix(segment_data.Y, segment_data.y_pred),
            }
        )

    return results
