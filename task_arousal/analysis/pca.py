"""
Module for performing PCA on fMRI data.
"""

from dataclasses import dataclass, field
from typing import List

import fbpca
import numpy as np

from scipy import linalg


def compute_variance_mask(
    X: np.ndarray, var_threshold_factor: float = 100.0
) -> np.ndarray:
    """
    Compute a boolean mask that excludes voxels with extreme temporal variance.

    Voxels whose variance exceeds ``var_threshold_factor`` times the median
    voxel variance are flagged as outliers (e.g. edge voxels, high-susceptibility
    regions) and excluded from the PCA.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    var_threshold_factor : float
        Multiplier applied to the median variance to set the exclusion threshold.
        Default 100.0.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_features,). True = keep voxel.
    """
    voxel_var = X.var(axis=0)
    threshold = var_threshold_factor * np.median(voxel_var)
    return voxel_var <= threshold


def restore_masked_voxels(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Reinsert excluded voxels back into a full-brain array.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_rows, n_masked_voxels), where the last axis indexes
        the voxels that were *kept* by ``mask``.
    mask : np.ndarray
        Boolean mask of shape (n_voxels,), True = voxel was included in ``data``.
    fill_value : float
        Value assigned to excluded voxels. Default ``np.nan``.

    Returns
    -------
    np.ndarray
        Array of shape (n_rows, n_voxels) with excluded voxels filled by
        ``fill_value``.
    """
    n_rows = data.shape[0]
    n_voxels = mask.shape[0]
    out = np.full((n_rows, n_voxels), fill_value, dtype=float)
    out[:, mask] = data
    return out


@dataclass
class PCAResults:
    """
    Class for storing results of PCA.

    Attributes
    ----------
    pc_scores: np.ndarray
        the PC scores of the principal components

    loadings: np.ndarray
        the loadings of the principal components

    explained_variance: np.ndarray
        the explained variance of the principal components

    U: np.ndarray
        left singular vectors from fbpca

    s: np.ndarray
        singular values from fbpca

    Va: np.ndarray
        right singular vectors from fbpca

    mean: np.ndarray
        temporal mean of the input data, computed and removed before decomposition,
        shape (n_features,). Required for reconstruction: X ≈ pc_scores @ Va + mean
    """

    pc_scores: np.ndarray
    loadings: np.ndarray
    explained_variance: np.ndarray
    U: np.ndarray
    s: np.ndarray
    Va: np.ndarray
    mean: np.ndarray = field(default_factory=lambda: np.array([]))


class PCA:
    """
    Principal Component Analysis (PCA) implementation using
    the fbpca library for efficient computation.
    """

    def __init__(
        self,
        n_components: int = 10,
        n_iter: int = 10,
    ):
        """Initialize PCA.

        Attributes
        ----------
           n_components : int
        """
        self.n_components = n_components
        self.n_iter = n_iter

    def decompose(self, X: np.ndarray) -> PCAResults:
        """
        Perform PCA on the input data matrix X. The temporal mean is computed
        and removed internally and stored in PCAResults.

        Parameters
        ----------
        X : np.ndarray
            The input data matrix of shape (n_samples, n_features).

        Returns
        -------
        PCAResults
            The results of the PCA decomposition.
        """
        # compute and remove temporal mean
        mean = X.mean(axis=0)
        X = X - mean
        # get number of observations
        n_samples = X.shape[0]
        # fbpca pca
        (U, s, Va) = fbpca.pca(X, k=self.n_components, n_iter=self.n_iter)
        # calc explained variance
        explained_variance_ = ((s**2) / (n_samples - 1)) / X.shape[1]
        # compute PC scores
        pc_scores = X @ Va.T
        # get loadings from eigenvectors
        loadings = Va.T @ np.diag(s)
        loadings /= np.sqrt(X.shape[0] - 1)
        return PCAResults(
            pc_scores=pc_scores,
            loadings=loadings,
            explained_variance=explained_variance_,
            U=U,
            s=s,
            Va=Va,
            mean=mean,
        )


@dataclass
class GroupPCAResults:
    """
    Class for storing results of GroupPCA.

    Attributes
    ----------
    pc_scores : np.ndarray
        Shared latent scores of shape (n_samples, n_components).

    loadings : np.ndarray
        Loadings of the group PCA components of shape (n_features_total, n_components).

    explained_variance : np.ndarray
        Explained variance of each group component, shape (n_components,).

    U : np.ndarray
        Left singular vectors from fbpca, shape (n_samples, n_components).

    s : np.ndarray
        Singular values from fbpca, shape (n_components,).

    Va : np.ndarray
        Right singular vectors from fbpca over the concatenated PC space,
        shape (n_components, n_individual_components_total).

    individual_projections : list of np.ndarray
        Encoder for each view in voxel space.
        individual_projections[i] has shape (n_components, n_voxels_i).
        Maps view i's voxels to the shared latent space:
            pc_scores_i ≈ (X_i - mean_i) @ individual_projections[i].T

    individual_embeddings : list of np.ndarray
        Decoder for each view in voxel space.
        individual_embeddings[i] has shape (n_voxels_i, n_components).
        Maps the shared latent space back to view i's voxels:
            X_i ≈ pc_scores @ individual_embeddings[i].T + mean_i
    """

    pc_scores: np.ndarray
    loadings: np.ndarray
    explained_variance: np.ndarray
    U: np.ndarray
    s: np.ndarray
    Va: np.ndarray
    individual_projections: List[np.ndarray] = field(default_factory=list)
    individual_embeddings: List[np.ndarray] = field(default_factory=list)


class GroupPCA:
    """
    Group PCA across multiple views (e.g. different fMRI echo acquisitions).

    Accepts a list of PCAResults from per-view individual PCAs. Concatenates
    the PC scores across views, runs a group PCA on the concatenated matrix,
    then estimates per-view encoder/decoder pairs in the original voxel space
    by composing the least-squares solution with each view's individual PCA
    components (Va).
    """

    def __init__(
        self,
        n_components: int = 10,
        n_iter: int = 10,
    ):
        """Initialize GroupPCA.

        Parameters
        ----------
        n_components : int
            Number of group components to extract.
        n_iter : int
            Number of power iterations for fbpca.
        """
        self.n_components = n_components
        self.n_iter = n_iter

    def decompose(self, pca_results: List[PCAResults]) -> GroupPCAResults:
        """
        Perform Group PCA on a list of per-view PCAResults.

        Parameters
        ----------
        pca_results : list of PCAResults
            One PCAResults per view (e.g. per echo). Each must have been fit
            on mean-centered data. The pc_scores field is used as input to the
            group PCA; Va is used to map projections back to voxel space.

        Returns
        -------
        GroupPCAResults
            Results of the group PCA, including per-view encoders and decoders
            in the original voxel space.
        """
        Xs = [r.pc_scores for r in pca_results]
        Va_list = [r.Va for r in pca_results]

        # concatenate PC scores across views: (n_samples, sum of n_individual_components)
        X_stack = np.hstack(Xs)
        n_samples = X_stack.shape[0]

        # group PCA on concatenated PC scores
        (U, s, Va) = fbpca.pca(X_stack, k=self.n_components, n_iter=self.n_iter)
        explained_variance = ((s**2) / (n_samples - 1)) / X_stack.shape[1]
        pc_scores = X_stack @ Va.T
        loadings = Va.T @ np.diag(s)
        loadings /= np.sqrt(n_samples - 1)

        # least-squares: find A_i (in PC space) such that Z @ A_i ≈ X̃_i
        # then compose with Va_i to get voxel-space encoder/decoder
        transformed_pinv = linalg.pinv(pc_scores)  # (n_components, n_samples)
        individual_projections = []
        individual_embeddings = []
        for X_pc, Va_i in zip(Xs, Va_list):
            # lstq_solution: (n_components, n_individual_components_i)
            lstq_solution = transformed_pinv @ X_pc
            # compose with Va_i to lift to voxel space
            # encoder: (n_components, n_voxels_i)
            encoder_voxel = np.linalg.pinv(lstq_solution).T @ Va_i
            # decoder: (n_voxels_i, n_components)
            decoder_voxel = (lstq_solution @ Va_i).T
            individual_projections.append(encoder_voxel)
            individual_embeddings.append(decoder_voxel)

        return GroupPCAResults(
            pc_scores=pc_scores,
            loadings=loadings,
            explained_variance=explained_variance,
            U=U,
            s=s,
            Va=Va,
            individual_projections=individual_projections,
            individual_embeddings=individual_embeddings,
        )
