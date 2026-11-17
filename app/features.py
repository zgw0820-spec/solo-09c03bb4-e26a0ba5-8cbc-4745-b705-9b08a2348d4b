"""Normalisation of user-supplied training input.

The API accepts raw fan vibration/current features and standardises each
column with statistics from the submitted sample (zero mean, unit variance);
constant columns become zero.  A bias column of ones is appended, so the
learned vector is ``[w_1 .. w_d, b]``.  Labels must be 0 (healthy) or 1
(shutdown warning).

The float64 design matrix is persisted, so any worker resumes on bit-identical
inputs.
"""
from __future__ import annotations

import numpy as np

MIN_RECORDS = 8
MAX_RECORDS = 64
MIN_EPOCHS = 2
MAX_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class InputError(ValueError):
    """Invalid training submission."""


def normalise(features: list[list[float]], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not (MIN_RECORDS <= len(features) <= MAX_RECORDS):
        raise InputError(f"submit between {MIN_RECORDS} and {MAX_RECORDS} records")
    if len(labels) != len(features):
        raise InputError("features and labels must contain the same number of records")
    if any(len(row) == 0 for row in features):
        raise InputError("each record needs at least one feature")
    if any(len(row) != len(features[0]) for row in features):
        raise InputError("every record must have the same number of features")

    try:
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError("features and labels must be numeric") from exc
    if matrix.ndim != 2:
        raise InputError("features must be a two-dimensional list of numbers")
    if not np.all(np.isfinite(matrix)):
        raise InputError("features must be finite numbers")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise InputError("labels must be 0 or 1")

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    constant = std < 1e-12
    scaled = (matrix - mean) / np.where(constant, 1.0, std)
    scaled[:, constant] = 0.0
    design = np.concatenate([scaled, np.ones((scaled.shape[0], 1), dtype=np.float64)], axis=1)
    return design, targets
