"""Smoke tests that validate the pipeline is installed and wired up
correctly, without requiring the (private, not bundled) EEG dataset or a GPU.

Run with:
    pytest tests/test_smoke.py -v
or directly:
    python tests/test_smoke.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_imports():
    """All core modules import cleanly without TensorFlow installed (catches
    missing/broken dependencies, and regressions where a CLI/data-path module
    accidentally pulls in TensorFlow at import time)."""
    import adhd_eeg.config  # noqa: F401
    import adhd_eeg.cv  # noqa: F401
    import adhd_eeg.features  # noqa: F401
    import adhd_eeg.resnets_utils  # noqa: F401
    import adhd_eeg.train  # noqa: F401  (import-only check)


def test_models_import():
    """``adhd_eeg.models`` requires TensorFlow; skipped with a clear reason
    when it isn't installed rather than failing."""
    pytest.importorskip("tensorflow", reason="TensorFlow is not installed")
    import adhd_eeg.models  # noqa: F401


def test_convert_to_one_hot():
    from adhd_eeg.resnets_utils import convert_to_one_hot

    y = np.array([0, 1, 1, 0])
    one_hot = convert_to_one_hot(y, 2).T
    assert one_hot.shape == (4, 2)
    assert (one_hot.argmax(axis=1) == y).all()


def test_cnn_model_forward_pass():
    """Build the CNN with the paper's input shape and run one dummy batch
    through it, verifying the architecture is wired correctly end to end."""
    pytest.importorskip("tensorflow", reason="TensorFlow is not installed")
    from adhd_eeg.config import IMAGE_SHAPE
    from adhd_eeg.models import cnn_model

    model = cnn_model(input_shape=IMAGE_SHAPE, classes=2)
    dummy_input = np.random.rand(2, *IMAGE_SHAPE).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 2)


def test_resnet_model_forward_pass():
    pytest.importorskip("tensorflow", reason="TensorFlow is not installed")
    from adhd_eeg.config import IMAGE_SHAPE
    from adhd_eeg.models import resnet_model

    model = resnet_model(input_shape=IMAGE_SHAPE, classes=2)
    dummy_input = np.random.rand(2, *IMAGE_SHAPE).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 2)


def test_resolve_data_dir_missing_gives_helpful_error(tmp_path):
    from adhd_eeg.config import resolve_data_dir

    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_data_dir(missing)


def test_resolve_data_dir_missing_subfolders_gives_helpful_error(tmp_path):
    from adhd_eeg.config import resolve_data_dir

    (tmp_path / "ADHD_part1").mkdir()
    with pytest.raises(FileNotFoundError, match="missing expected"):
        resolve_data_dir(tmp_path)


def test_resolve_data_dir_success(tmp_path):
    from adhd_eeg.config import REQUIRED_SUBDIRS, resolve_data_dir

    for sub in REQUIRED_SUBDIRS:
        (tmp_path / sub).mkdir()
    assert resolve_data_dir(tmp_path) == tmp_path


def test_cli_fails_clearly_without_dataset(tmp_path):
    """`python -m adhd_eeg.train` on a fresh machine (no dataset) should exit
    non-zero with a clear, actionable error rather than crashing deep inside
    the pipeline."""
    result = subprocess.run(
        [sys.executable, "-m", "adhd_eeg.train", "--data-dir", str(tmp_path / "missing")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
