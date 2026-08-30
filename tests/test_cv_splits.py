"""Tests for fold construction, including the subject-leakage regression test.

These need neither TensorFlow nor the EEG dataset: fold construction is pure
bookkeeping over a small synthetic table.
"""
from __future__ import annotations

import pandas as pd
import pytest

from adhd_eeg.cv import build_folds, segment_table, subjects_in

SEGMENT_SIZE = 20


def make_full_data(n_subjects: int = 20, segments_per_subject: int = 4) -> pd.DataFrame:
    """A table shaped like the real one: each subject owns several segments.

    Labels alternate by subject, mirroring the ADHD / control split.
    """
    rows = []
    segment_id = 0
    for person in range(1, n_subjects + 1):
        label = person % 2
        for _ in range(segments_per_subject):
            for _ in range(SEGMENT_SIZE):
                rows.append({"Y": label, "person": person, "segment": segment_id})
            segment_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def full_data():
    return make_full_data()


class TestSubjectLeakage:
    """The published pipeline splits segments, not subjects."""

    def test_segment_split_puts_the_same_subject_on_both_sides(self, full_data):
        folds = build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by="segment"
        )
        overlaps = [
            subjects_in(full_data, train) & subjects_in(full_data, test)
            for train, test in folds
        ]
        assert any(overlaps), "expected subject overlap under the segment strategy"

    def test_subject_split_has_no_overlap_in_any_fold(self, full_data):
        folds = build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by="subject"
        )
        for index, (train, test) in enumerate(folds):
            overlap = subjects_in(full_data, train) & subjects_in(full_data, test)
            assert not overlap, f"fold {index} leaks subjects {sorted(overlap)}"

    def test_segment_strategy_leaks_most_test_subjects(self, full_data):
        """Quantifies the leak: nearly every test subject is also trained on."""
        folds = build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by="segment"
        )
        leaked, total = 0, 0
        for train, test in folds:
            test_subjects = subjects_in(full_data, test)
            total += len(test_subjects)
            leaked += len(test_subjects & subjects_in(full_data, train))
        assert leaked / total > 0.5


class TestFoldStructure:
    @pytest.mark.parametrize("strategy", ["segment", "subject"])
    def test_every_fold_has_train_and_test_segments(self, full_data, strategy):
        for train, test in build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by=strategy
        ):
            assert train and test

    @pytest.mark.parametrize("strategy", ["segment", "subject"])
    def test_train_and_test_segments_are_disjoint(self, full_data, strategy):
        for train, test in build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by=strategy
        ):
            assert not (set(train) & set(test))

    @pytest.mark.parametrize("strategy", ["segment", "subject"])
    def test_requested_number_of_folds_is_returned(self, full_data, strategy):
        assert len(build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by=strategy
        )) == 5

    def test_every_complete_segment_is_tested_exactly_once(self, full_data):
        folds = build_folds(
            full_data, segment_size=SEGMENT_SIZE, n_splits=5, split_by="subject"
        )
        tested = [segment for _, test in folds for segment in test]
        assert len(tested) == len(set(tested))
        assert set(tested) == set(full_data.segment.unique())

    def test_partial_segments_always_go_to_training(self):
        """Undersized segments are training-only, matching the notebook."""
        data = make_full_data(n_subjects=10, segments_per_subject=3)
        partial_id = data.segment.max() + 1
        stub = pd.DataFrame(
            [{"Y": 1, "person": 1, "segment": partial_id}] * 3  # < SEGMENT_SIZE
        )
        data = pd.concat([data, stub], ignore_index=True)

        for train, test in build_folds(
            data, segment_size=SEGMENT_SIZE, n_splits=5, split_by="subject"
        ):
            assert partial_id in train
            assert partial_id not in test

    def test_is_reproducible_for_a_fixed_seed(self, full_data):
        first = build_folds(full_data, segment_size=SEGMENT_SIZE, n_splits=5,
                            random_state=3, split_by="subject")
        second = build_folds(full_data, segment_size=SEGMENT_SIZE, n_splits=5,
                             random_state=3, split_by="subject")
        assert [sorted(t) for _, t in first] == [sorted(t) for _, t in second]


class TestSegmentTable:
    def test_one_row_per_segment(self, full_data):
        table = segment_table(full_data)
        assert len(table) == full_data.segment.nunique()

    def test_rejects_a_segment_spanning_two_subjects(self, full_data):
        corrupted = full_data.copy()
        corrupted.loc[corrupted.index[:5], "person"] = 999
        with pytest.raises(ValueError, match="more than one subject"):
            segment_table(corrupted)

    def test_rejects_a_segment_with_two_labels(self, full_data):
        corrupted = full_data.copy()
        corrupted.loc[corrupted.index[:5], "Y"] = 1 - corrupted.loc[corrupted.index[0], "Y"]
        with pytest.raises(ValueError, match="more than one label"):
            segment_table(corrupted)


class TestValidation:
    def test_unknown_strategy_is_rejected(self, full_data):
        with pytest.raises(ValueError, match="unknown split_by"):
            build_folds(full_data, segment_size=SEGMENT_SIZE, split_by="telepathy")

    def test_more_folds_than_subjects_is_rejected(self):
        data = make_full_data(n_subjects=4, segments_per_subject=3)
        with pytest.raises(ValueError, match="subject-wise folds"):
            build_folds(data, segment_size=SEGMENT_SIZE, n_splits=10, split_by="subject")


class TestSubjectsIn:
    def test_returns_the_covering_subjects(self, full_data):
        segments = full_data.loc[full_data.person == 1, "segment"].unique()
        assert subjects_in(full_data, segments) == {1}

    def test_empty_selection_gives_an_empty_set(self, full_data):
        assert subjects_in(full_data, []) == set()
