"""
Simple seed-based functional connectivity (FC) analysis for fMRI data.
"""

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA


@dataclass
class SeedFCResults:
    """
    Class for storing seed-based functional connectivity results.

    Attributes
    ----------
    connectivity_map : np.ndarray
        Voxelwise FC map of shape (n_voxels,).
    seed_timecourse : np.ndarray
        Seed time course used to compute the FC map.
    roi_mask : np.ndarray | None
        Optional ROI mask used to derive the seed time course.
    roi_strategy : Literal['mean', 'pca'] | None
        Strategy used to derive the seed time course from ``roi_mask``.
    metric : Literal['correlation', 'covariance']
        Connectivity metric used for estimation.
    fisher_transformed : bool
        Whether Fisher's $z$ transform was applied to the FC map.
    seed_source : Literal['timecourse', 'roi_mask']
        Indicates whether the seed was supplied directly or derived from an ROI
        mask.
    """

    connectivity_map: np.ndarray
    seed_timecourse: np.ndarray
    roi_mask: np.ndarray | None
    roi_strategy: Literal["mean", "pca"] | None
    metric: Literal["correlation", "covariance"]
    fisher_transformed: bool
    seed_source: Literal["timecourse", "roi_mask"]


class SeedBasedFC:
    """
    Simple seed-based functional connectivity analysis on
    ``(n_timepoints, n_voxels)`` fMRI matrices.

    This module computes a voxelwise FC map between a seed time course and all
    voxel time courses in a functional run.
    """

    def __init__(
        self,
        metric: Literal["correlation", "covariance"] = "correlation",
        fisher_transform: bool = False,
        roi_strategy: Literal["mean", "pca"] = "mean",
        eps: float = 1e-12,
    ):
        self.metric = metric
        self.fisher_transform = bool(fisher_transform)
        self.roi_strategy = roi_strategy
        self.eps = float(eps)

        if self.metric not in {"correlation", "covariance"}:
            raise ValueError("metric must be one of {'correlation', 'covariance'}")
        if self.metric != "correlation" and self.fisher_transform:
            raise ValueError(
                "fisher_transform is only defined for correlation-based FC maps"
            )
        if self.roi_strategy not in {"mean", "pca"}:
            raise ValueError("roi_strategy must be one of {'mean', 'pca'}")

    def _validate_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_timepoints, n_voxels), got shape {X.shape}"
            )

        return X

    def _validate_seed_timecourse(
        self, X: np.ndarray, seed_timecourse: np.ndarray
    ) -> np.ndarray:
        seed_timecourse = np.asarray(seed_timecourse, dtype=float)

        if seed_timecourse.ndim == 2:
            if seed_timecourse.shape[1] != 1:
                raise ValueError(
                    "seed_timecourse must be 1D or 2D with shape (n_timepoints, 1)"
                )
            seed_timecourse = seed_timecourse[:, 0]
        elif seed_timecourse.ndim != 1:
            raise ValueError(
                "seed_timecourse must be 1D or 2D with shape (n_timepoints, 1)"
            )

        if X.shape[0] != seed_timecourse.shape[0]:
            raise ValueError(
                "Time dimension mismatch: "
                f"X has T={X.shape[0]}, seed_timecourse has T={seed_timecourse.shape[0]}"
            )

        return seed_timecourse

    def _validate_roi_mask(self, n_voxels: int, roi_mask: np.ndarray) -> np.ndarray:
        roi_mask = np.asarray(roi_mask).astype(bool)
        if roi_mask.ndim != 1:
            raise ValueError(f"roi_mask must be 1D, got shape {roi_mask.shape}")
        if roi_mask.shape[0] != n_voxels:
            raise ValueError(
                f"roi_mask length ({roi_mask.shape[0]}) must match n_voxels ({n_voxels})"
            )
        if int(np.sum(roi_mask)) == 0:
            raise ValueError("roi_mask contains no True voxels")
        return roi_mask

    def _resolve_seed(
        self,
        X: np.ndarray,
        seed_timecourse: np.ndarray | None,
        roi_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, Literal["timecourse", "roi_mask"]]:
        if seed_timecourse is None and roi_mask is None:
            raise ValueError(
                "Provide either seed_timecourse or roi_mask to compute seed-based FC"
            )
        if seed_timecourse is not None and roi_mask is not None:
            raise ValueError(
                "Provide only one of seed_timecourse or roi_mask, not both"
            )

        if seed_timecourse is not None:
            return (
                self._validate_seed_timecourse(X, seed_timecourse),
                None,
                "timecourse",
            )
        # ensure roi mask is not None
        if roi_mask is None:
            raise ValueError("roi_mask cannot be None when seed_timecourse is None")
        mask = self._validate_roi_mask(X.shape[1], roi_mask)
        roi_data = X[:, mask]

        if self.roi_strategy == "mean":
            seed = roi_data.mean(axis=1)
        elif self.roi_strategy == "pca":
            pca = SklearnPCA(n_components=1)
            seed = pca.fit_transform(roi_data)[:, 0]
            if np.corrcoef(seed, roi_data.mean(axis=1))[0, 1] < 0:
                seed = -seed
        else:
            raise ValueError(f"Unknown roi_strategy={self.roi_strategy!r}")

        return seed, mask, "roi_mask"

    def _compute_correlation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_centered = X - X.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()

        x_norm = np.sqrt(np.sum(X_centered**2, axis=0))
        y_norm = float(np.sqrt(np.sum(y_centered**2)))

        if y_norm <= self.eps:
            raise ValueError(
                "seed_timecourse has near-zero variance; correlation is undefined"
            )

        denom = x_norm * y_norm
        valid = denom > self.eps

        if not np.all(valid):
            warnings.warn(
                "Some voxel time courses have near-zero variance; returning NaN for those voxels.",
                stacklevel=2,
            )

        corr = np.full(X.shape[1], np.nan, dtype=float)
        corr[valid] = (X_centered[:, valid].T @ y_centered) / denom[valid]
        corr = np.clip(corr, -1.0, 1.0)
        return corr

    def _compute_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_centered = X - X.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()
        dof = max(X.shape[0] - 1, 1)
        return (X_centered.T @ y_centered) / dof

    def compute(
        self,
        X: np.ndarray,
        seed_timecourse: np.ndarray | None = None,
        roi_mask: np.ndarray | None = None,
    ) -> SeedFCResults:
        """
        Compute a simple seed-based FC map.

        Parameters
        ----------
        X : np.ndarray
            fMRI data matrix of shape ``(n_timepoints, n_voxels)``.
        seed_timecourse : np.ndarray | None
            Seed time course of shape ``(n_timepoints,)`` or
            ``(n_timepoints, 1)``.
        roi_mask : np.ndarray | None
            Optional 1D boolean ROI mask of shape ``(n_voxels,)``. If passed,
            the seed time course is computed as the mean across masked voxels.

        Returns
        -------
        SeedFCResults
            FC results containing the voxelwise connectivity map.
        """
        X = self._validate_inputs(X)
        seed_timecourse, resolved_roi_mask, seed_source = self._resolve_seed(
            X=X,
            seed_timecourse=seed_timecourse,
            roi_mask=roi_mask,
        )

        if self.metric == "correlation":
            fc_map = self._compute_correlation(X, seed_timecourse)
            if self.fisher_transform:
                finite = np.isfinite(fc_map)
                fc_map[finite] = np.arctanh(
                    np.clip(fc_map[finite], -1.0 + self.eps, 1.0 - self.eps)
                )
        elif self.metric == "covariance":
            fc_map = self._compute_covariance(X, seed_timecourse)
        else:
            raise ValueError(f"Unknown metric={self.metric!r}")

        return SeedFCResults(
            connectivity_map=fc_map,
            seed_timecourse=np.array(seed_timecourse, dtype=float, copy=True),
            roi_mask=None if resolved_roi_mask is None else resolved_roi_mask.copy(),
            roi_strategy=None if seed_source == "timecourse" else self.roi_strategy,  # type: ignore[arg-type]
            metric=self.metric,
            fisher_transformed=self.fisher_transform,
            seed_source=seed_source,
        )


def seed_based_fc(
    X: np.ndarray,
    seed_timecourse: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    metric: Literal["correlation", "covariance"] = "correlation",
    fisher_transform: bool = False,
    roi_strategy: Literal["mean", "pca"] = "mean",
    eps: float = 1e-12,
) -> SeedFCResults:
    """
    Convenience wrapper for simple seed-based FC.

    Parameters
    ----------
    X : np.ndarray
        fMRI data matrix of shape ``(n_timepoints, n_voxels)``.
    seed_timecourse : np.ndarray | None
        Seed time course of shape ``(n_timepoints,)`` or
        ``(n_timepoints, 1)``.
    roi_mask : np.ndarray | None
        Optional 1D boolean ROI mask of shape ``(n_voxels,)``. If passed,
        the seed time course is computed as the mean across masked voxels.
    roi_strategy : Literal['mean', 'pca']
        Strategy used to derive the seed time course from ``roi_mask``.
        ``'mean'`` averages masked voxels and ``'pca'`` extracts the first
        principal component.
    metric : Literal['correlation', 'covariance']
        Connectivity metric to compute.
    fisher_transform : bool
        Whether to apply Fisher's $z$ transform to correlation maps.
    eps : float
        Numerical stability constant.

    Returns
    -------
    SeedFCResults
        FC results containing the voxelwise connectivity map.
    """
    model = SeedBasedFC(
        metric=metric,
        fisher_transform=fisher_transform,
        roi_strategy=roi_strategy,
        eps=eps,
    )
    return model.compute(X=X, seed_timecourse=seed_timecourse, roi_mask=roi_mask)
