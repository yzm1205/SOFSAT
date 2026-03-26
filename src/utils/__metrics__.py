import sys

import numpy as np
from numpy.typing import NDArray

from .__projection__ import _normalize
from .__projection__ import _a_dot_b as a_dot_b
from .__projection__ import _cosine_similarity as cosine_similarity


def l1_distance(x: NDArray, y: NDArray) -> NDArray:
    return np.linalg.norm((x - y), 1, axis=-1, keepdims=True)


def l2_distance(x: NDArray, y: NDArray) -> NDArray:
    return np.linalg.norm((x - y), 2, axis=-1, keepdims=True)


def nsed_distance(x: NDArray, y: NDArray) -> NDArray:
    def compute_var(xx: NDArray) -> NDArray:
        return np.var(xx, axis=-1, keepdims=True)

    return 0.5 * compute_var((x - y)) / (compute_var(x) + compute_var(y))


def normalized_l1_distance(x: NDArray, y: NDArray) -> NDArray:
    x = _normalize(x)
    y = _normalize(y)
    return l1_distance(x, y)


def normalized_l2_distance(x: NDArray, y: NDArray) -> NDArray:
    x = _normalize(x)
    y = _normalize(y)
    return l2_distance(x, y)


def mahalanobis_distance(x: NDArray, y: NDArray, inv_cov_matrix: NDArray) -> NDArray:
    """
    # TODO: Compute the covariance matrix of the entire data beforehand and pass it here.

    Compute the Mahalanobis distance between two batches of vectors.

    Parameters:
    x (numpy.ndarray): A batch of vectors of shape (B, D).
    y (numpy.ndarray): Another batch of vectors of shape (B, D).
    inv_cov_matrix (numpy.ndarray): Covariance matrix of the target distribution (D, D)

    Returns:
    numpy.ndarray: Mahalanobis distances of shape (B,).
    """
    # Ensure x and y have the same shape
    assert x.shape == y.shape, "The shapes of x and y must be the same"

    # This is how you compute inverse covariance matrix
    # # Compute the mean vector
    # mean = np.mean(np.vstack([x, y]), axis=0)  # np.mean([2B x D], axis=0)

    # # Center the data
    # x_centered = x - mean
    # y_centered = y - mean

    # # Compute the covariance matrix
    # cov_matrix = np.cov(np.vstack([x_centered, y_centered]).T)  # D x D

    # # Compute the inverse of the covariance matrix
    # cov_matrix_inv = np.linalg.inv(cov_matrix)  # D x D

    # Compute the Mahalanobis distance
    diff = x - y
    out = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
    return out


METRICS_MAPPING = {
    # "mahalanobis_distance": mahalanobis_distance,
    # "normalized_l1": normalized_l1_distance,
    # "normalized_l2": normalized_l2_distance,
    # "l1": l1_distance,
    # "l2": l2_distance,
    # "nsed": nsed_distance,
    "cos": cosine_similarity,
    # "dot": a_dot_b,
}


if __name__ == "__main__":
    print(METRICS_MAPPING.keys())
