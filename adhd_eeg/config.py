"""Dataset path resolution and configuration.

The original notebook hardcoded Google Drive paths (e.g.
``/content/drive/MyDrive/adhd_dataset``). This module replaces that with a
configurable path so the pipeline can run on any machine.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Environment variable used to locate the dataset when no --data-dir is given.
DATA_DIR_ENV_VAR = "ADHD_DATA_DIR"

#: Default location, relative to the current working directory.
DEFAULT_DATA_DIR = Path("data/adhd_dataset")

#: Subfolders the dataset must contain, each holding EEGLAB ``.set`` files.
REQUIRED_SUBDIRS = ("ADHD_part1", "ADHD_part2", "Control_part1", "Control_part2")

#: Number of consecutive epochs grouped into one subject-level "segment"
#: (matches ``con = 20`` in the original notebook).
DEFAULT_SEGMENT_SIZE = 20

#: Image / model input shape produced by the feature extraction pipeline.
IMAGE_SHAPE = (19, 256, 3)


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve and validate the EEG dataset directory.

    Resolution order: explicit ``data_dir`` argument, then the
    ``ADHD_DATA_DIR`` environment variable, then ``./data/adhd_dataset``.

    Raises
    ------
    FileNotFoundError
        With a message explaining how to obtain/place the dataset, if the
        directory (or one of its required subfolders) is missing.
    """
    if data_dir is not None:
        path = Path(data_dir)
    else:
        path = Path(os.environ.get(DATA_DIR_ENV_VAR, DEFAULT_DATA_DIR))

    if not path.exists():
        raise FileNotFoundError(
            f"EEG dataset directory not found at '{path}'.\n"
            "This repository does not bundle the dataset. Obtain the ADHD / "
            "control EEGLAB recordings used in the paper and place them so "
            f"that '{path}' contains the subfolders: {', '.join(REQUIRED_SUBDIRS)} "
            "(each holding '.set'/'.fdt' files). Alternatively, point at an "
            f"existing copy via the {DATA_DIR_ENV_VAR} environment variable "
            "or the --data-dir command line flag."
        )

    missing = [d for d in REQUIRED_SUBDIRS if not (path / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset directory '{path}' exists but is missing expected "
            f"subfolder(s): {', '.join(missing)}. Expected layout:\n"
            f"  {path}/ADHD_part1/*.set\n"
            f"  {path}/ADHD_part2/*.set\n"
            f"  {path}/Control_part1/*.set\n"
            f"  {path}/Control_part2/*.set"
        )

    return path
