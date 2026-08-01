"""Small helper(s) the notebook imported from an external ``resnets_utils``
module that was never included in the repository.

Only the one helper actually used by ``ADHD.ipynb`` (``convert_to_one_hot``)
is reimplemented here, as a plain NumPy one-hot encoder.
"""
from __future__ import annotations

import numpy as np


def convert_to_one_hot(Y: np.ndarray, C: int) -> np.ndarray:
    """One-hot encode integer labels.

    Parameters
    ----------
    Y: array of shape (m,) with integer class labels in ``[0, C)``.
    C: number of classes.

    Returns
    -------
    Array of shape (C, m). Callers in this codebase transpose the result
    to get the more common (m, C) shape.
    """
    return np.eye(C)[Y.reshape(-1)].T
