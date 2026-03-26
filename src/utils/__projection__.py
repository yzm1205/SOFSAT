import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def _a_dot_b(vec_a: NDArray, vec_b: NDArray) -> NDArray:
    """Computes a.b between batch of two matrices of same shape (BxD). Output dim - Bx1"""
    return (vec_a * vec_b).sum(axis=1, keepdims=True)


def _norm(vec: NDArray) -> NDArray:
    return np.sqrt(_a_dot_b(vec, vec))


def _cosine_similarity(vec_a: NDArray, vec_b: NDArray) -> NDArray:
    a_b = _a_dot_b(vec_a, vec_b)
    return np.clip(a_b / (_norm(vec_a) * _norm(vec_b)), -1.0, 1.0)


def _normalize(vec: NDArray) -> NDArray:
    out = vec / _norm(vec)  # unit vectors
    try:
        assert np.mean(np.abs(_norm(out) - 1)) < 1e-6
    except Exception as e:
        aa = 1
        print("Done")
    return out


def _compute_projection(
    vec: NDArray, base_vec: NDArray, normalized: bool = False
) -> NDArray:
    """Computes the projection of a vector A on Vec B"""
    normalised_base_vec = base_vec
    if not normalized:
        normalised_base_vec = _normalize(base_vec)
    x_y = _a_dot_b(vec, normalised_base_vec)
    return normalised_base_vec * x_y  # B *((A.B))  where B is normalized vec


def _compute_basis_vectors(vec_a: NDArray, vec_b: NDArray) -> Tuple[NDArray, NDArray]:
    """Computes Basis Vectors (orthonormal) of the plane"""
    basis_b = _normalize(vec_b)
    basis_a = _normalize(vec_a - _compute_projection(vec_a, basis_b, normalized=True))
    return basis_a, basis_b


def _compute_projection_on_plane(
    vec: NDArray, plane_vec_a: NDArray, plane_vec_b: NDArray
) -> NDArray:
    basis_a, basis_b = _compute_basis_vectors(plane_vec_a, plane_vec_b)
    vec_proj_on_a = _compute_projection(vec, basis_a)
    vec_proj_on_b = _compute_projection(vec, basis_b)
    return vec_proj_on_a + vec_proj_on_b


def _cos_angle(cos_theta: NDArray, degree: bool = True) -> NDArray:
    out = np.arccos(cos_theta)
    if degree:
        out = np.degrees(out)
    return out


def _tanimoto_sim(x: NDArray, y: NDArray) -> NDArray:
    """A.B/(A^2 + B^2 - A.B)"""
    ts = _a_dot_b(x, y) / (_a_dot_b(x, x) + _a_dot_b(y, y) - _a_dot_b(x, y))
    return ts


def compute_angles(
    vec: NDArray, plane_vec_a: NDArray, plane_vec_b: NDArray, degree: bool = False
) -> NDArray:
    """Computes the angles between the projection vec (on the plane of a and b) and a and b respectively"""

    def _to_list(arr: NDArray) -> list:
        return arr.squeeze().tolist()

    assert len(vec.shape) == 2, vec.shape == plane_vec_a.shape == plane_vec_b.shape
    proj = _compute_projection_on_plane(vec, plane_vec_a, plane_vec_b)

    # Compute Angles
    proj_a_cos = _cos_angle(_cosine_similarity(proj, plane_vec_a), degree=True)
    proj_b_cos = _cos_angle(_cosine_similarity(proj, plane_vec_b), degree=True)
    a_b_cos = _cos_angle(_cosine_similarity(plane_vec_a, plane_vec_b), degree=True)
    diffs = np.abs(a_b_cos - (proj_a_cos + proj_b_cos))
    in_out = np.where(diffs < 1, "In", "Out")

    # Tanimoto Similarity

    return np.stack(
        [
            in_out,
            a_b_cos,
            proj_a_cos,
            proj_b_cos,
            _norm(plane_vec_a),
            _norm(plane_vec_b),
            _norm(proj),
            _norm(vec),
            _tanimoto_sim(proj, plane_vec_a),
            _tanimoto_sim(proj, plane_vec_b),
            _tanimoto_sim(plane_vec_a, plane_vec_b),
        ],
        axis=-1,
    ).squeeze()
