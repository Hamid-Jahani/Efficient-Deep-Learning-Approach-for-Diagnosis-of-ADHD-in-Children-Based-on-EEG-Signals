"""EEG -> Morlet-wavelet RGB image feature extraction.

This is a faithful extraction of the preprocessing pipeline in
``ADHD.ipynb``: each EEGLAB recording is split into fixed-length overlapping
epochs, transformed with a Morlet wavelet into theta / alpha / beta-gamma
power bands, and stacked into one RGB image per epoch.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Tuple

import mne
import numpy as np

#: Frequency bins passed to the Morlet transform (matches the notebook).
FREQUENCIES = np.arange(4, 40, 2)

EPOCH_DURATION = 2.0
EPOCH_OVERLAP = 1.5
N_CYCLES = 5

#: (subfolders, label) — ADHD = 1, control = 0.
GROUPS = (
    (("ADHD_part1", "ADHD_part2"), 1),
    (("Control_part1", "Control_part2"), 0),
)


def _band_power_frame(power, low: float, high: float):
    df = power.to_data_frame()
    band = df[(low <= df["freq"]) & (df["freq"] <= high)].drop(columns=["condition"])
    return band.groupby(["epoch", "time"]).agg("mean")


def recording_to_images(set_file: str) -> List[np.ndarray]:
    """Convert a single EEGLAB ``.set`` recording into a list of RGB images,
    one per fixed-length epoch."""
    raw = mne.io.read_raw_eeglab(set_file, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DURATION, preload=False, overlap=EPOCH_OVERLAP
    )
    power = mne.time_frequency.tfr_morlet(
        epochs,
        n_cycles=N_CYCLES,
        return_itc=False,
        freqs=FREQUENCIES,
        average=False,
        n_jobs=-1,
        verbose=False,
    )

    theta = _band_power_frame(power, 4, 8)
    alpha = _band_power_frame(power, 8, 12)
    beta_gamma = _band_power_frame(power, 12, 40)

    n_epochs = int(np.max(theta.reset_index().epoch)) + 1
    images = []
    for j in range(n_epochs):
        red = np.array(theta.loc[j].drop(columns=["freq"]).T)
        green = np.array(alpha.loc[j].drop(columns=["freq"]).T)
        blue = np.array(beta_gamma.loc[j].drop(columns=["freq"]).T)
        image = np.concatenate(
            (red[:, :, np.newaxis], green[:, :, np.newaxis], blue[:, :, np.newaxis]),
            axis=2,
        )
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        images.append(image)
    return images


def _find_set_files(data_dir: Path, subfolders: Iterable[str]) -> List[str]:
    files: List[str] = []
    for sub in subfolders:
        files.extend(sorted(glob.glob(str(data_dir / sub / "*.set"))))
    return files


def build_dataset(
    data_dir: Path, segment_size: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Build the full (images, labels, subjects, segments) arrays used for
    training, mirroring cells 5-14 of ``ADHD.ipynb``.

    Parameters
    ----------
    data_dir:
        Directory containing ``ADHD_part1``, ``ADHD_part2``, ``Control_part1``
        and ``Control_part2`` subfolders of EEGLAB ``.set`` files.
    segment_size:
        Number of consecutive epochs grouped into one subject-level
        "segment" (matches ``con = 20`` in the original notebook).

    Returns
    -------
    images: np.ndarray, shape (n_epochs, 19, 256, 3)
    labels: np.ndarray, shape (n_epochs,) -- 1 = ADHD, 0 = control
    subjects: list[int] -- 1-based subject id per epoch
    segments: list[int] -- segment id per epoch, reset at each subject boundary
    """
    data_dir = Path(data_dir)

    images: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[int] = []
    segments: List[int] = []

    person = 0
    segment_variable = 0
    condition = 0

    for subfolders, label in GROUPS:
        for set_file in _find_set_files(data_dir, subfolders):
            person += 1
            epoch_images = recording_to_images(set_file)
            n_epochs = len(epoch_images)
            for j, image in enumerate(epoch_images):
                images.append(image)
                labels.append(label)
                subjects.append(person)
                segments.append(segment_variable)
                condition += 1
                if condition >= segment_size:
                    segment_variable += 1
                    condition = 0
                if condition <= segment_size and j == n_epochs - 1:
                    segment_variable += 1
                    condition = 0

    if not images:
        raise RuntimeError(
            f"No '.set' files were found under '{data_dir}' in any of "
            f"{[g for group, _ in GROUPS for g in group]}. Check that the "
            "dataset was placed correctly."
        )

    return np.stack(images), np.array(labels), subjects, segments
