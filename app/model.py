"""Standard logistic regression: sigmoid, cross-entropy, SGD + momentum.

The arithmetic is deliberately fixed and float64 so the same sequence of
updates can be replayed from a checkpoint.  A restart therefore continues from
the exact weight and velocity vectors the previous worker committed.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable form: never exponentiate a large positive number.
    out = np.empty_like(z, dtype=np.float64)
    non_negative = z >= 0
    out[non_negative] = 1.0 / (1.0 + np.exp(-z[non_negative]))
    exp_z = np.exp(z[~non_negative])
    out[~non_negative] = exp_z / (1.0 + exp_z)
    return out


def batch_loss(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary cross-entropy of one batch."""
    logits = features @ weights
    probabilities = sigmoid(logits)
    eps = 1e-12
    positive = labels * np.log(probabilities + eps)
    negative = (1.0 - labels) * np.log(1.0 - probabilities + eps)
    return float(np.mean(-(positive + negative)))


def sgd_step(
    weights: np.ndarray,
    velocity: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    momentum: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One mini-batch update.

    Returns the new weights, the new momentum velocity and the loss measured on
    the batch *before* the update (so the reported loss corresponds to the
    parameters that were used to compute it).  The caller commits the resulting
    weights alongside that loss, keeping the checkpoint internally consistent.
    """
    probabilities = sigmoid(features @ weights)
    loss = batch_loss(weights, features, labels)
    gradient = features.T @ (probabilities - labels) / features.shape[0]
    velocity = momentum * velocity - learning_rate * gradient
    weights = weights + velocity
    return weights, velocity, loss


def dataset_loss(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    return batch_loss(weights, features, labels)
