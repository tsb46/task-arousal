"""Quadratic-penalty helpers for structured penalized least-squares models."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from scipy.sparse import block_diag, csc_matrix, eye
from scipy.sparse.linalg import splu


def as_csc(matrix: Any) -> csc_matrix:
    """Convert an array-like matrix to CSC sparse format."""
    if isinstance(matrix, csc_matrix):
        return matrix
    return csc_matrix(matrix)


def zero_penalty(size: int) -> csc_matrix:
    """Create an all-zero quadratic penalty matrix."""
    return csc_matrix((size, size), dtype=float)


def second_difference_matrix(size: int) -> np.ndarray:
    """Construct the second-difference operator for a one-dimensional axis."""
    if size < 3:
        return np.zeros((0, size), dtype=float)
    diff = np.zeros((size - 2, size), dtype=float)
    for idx in range(size - 2):
        # Apply [1, -2, 1] to neighboring entries so D @ beta measures discrete
        # curvature. Penalizing D.T @ D therefore discourages rapid local bending
        # without forcing the entire vector to be constant.
        diff[idx, idx : idx + 3] = [1.0, -2.0, 1.0]
    return diff


def ridge_penalty(size: int, weight: float) -> csc_matrix:
    """Construct a ridge penalty matrix."""
    if weight < 0:
        raise ValueError("ridge penalty weight must be non-negative")
    if weight == 0:
        return zero_penalty(size)
    return weight * eye(size, format="csc")


def second_difference_penalty(size: int, weight: float) -> csc_matrix:
    """Construct a quadratic penalty from a second-difference operator."""
    if weight < 0:
        raise ValueError("second-difference penalty weight must be non-negative")
    operator = second_difference_matrix(size)
    if weight == 0 or operator.shape[0] == 0:
        return zero_penalty(size)
    operator_csc = csc_matrix(operator)
    # The quadratic form beta.T @ (D.T D) @ beta equals ||D beta||^2, which is the
    # standard discrete roughness penalty used throughout the TE model.
    return weight * (operator_csc.T @ operator_csc)


def block_diagonal_penalty(blocks: Iterable[Any]) -> csc_matrix:
    """Assemble a block-diagonal penalty matrix from individual penalty blocks."""
    block_list = [as_csc(block) for block in blocks]
    if len(block_list) == 0:
        return csc_matrix((0, 0), dtype=float)
    return as_csc(block_diag(block_list, format="csc"))


class QuadraticPenaltySolver:
    """Factorize and solve penalized normal equations.

    The system solved is

    ``(G + P) beta = rhs``

    where ``G`` is the normal-equation matrix and ``P`` is a positive semidefinite
    quadratic penalty.
    """

    def __init__(
        self,
        normal_matrix: Any,
        penalty: Any | None = None,
    ):
        self.normal_matrix = as_csc(normal_matrix)
        normal_rows, normal_cols = self.normal_matrix.get_shape()
        if penalty is None:
            self.penalty = zero_penalty(normal_rows)
        else:
            self.penalty = as_csc(penalty)
        penalty_shape = self.penalty.get_shape()
        normal_shape = (normal_rows, normal_cols)
        if normal_shape != penalty_shape:
            raise ValueError(
                "normal_matrix and penalty must have the same shape, got "
                f"{normal_shape} and {penalty_shape}."
            )
        # The model-specific code constructs G from data and P from prior smoothness
        # assumptions, then this class handles the generic penalized solve
        # (G + P) beta = rhs.
        self.system_matrix = self.normal_matrix + self.penalty
        self._solver = splu(self.system_matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the penalized linear system for one or many right-hand sides."""
        rhs_array = np.asarray(rhs, dtype=float)
        system_rows, _ = self.system_matrix.get_shape()
        if rhs_array.shape[0] != system_rows:
            raise ValueError(
                "rhs has incompatible leading dimension. Expected "
                f"{system_rows}, got {rhs_array.shape[0]}."
            )
        return np.asarray(self._solver.solve(rhs_array), dtype=float)
