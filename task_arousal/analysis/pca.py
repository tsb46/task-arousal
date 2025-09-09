"""
Module for performing PCA on fMRI data.
"""

from dataclasses import dataclass

import fbpca
import numpy as np


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
    """

    pc_scores: np.ndarray
    loadings: np.ndarray
    explained_variance: np.ndarray

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
        Perform PCA on the input data matrix X.

        Parameters
        ----------
        X : np.ndarray
            The input data matrix of shape (n_samples, n_features).
        
        Returns
        -------
        PCAResults
            The results of the PCA decomposition.
        """
        # get number of observations
        n_samples = X.shape[0]
        # fbpca pca
        (U, s, Va) = fbpca.pca(X, k=self.n_components, n_iter=self.n_iter)
        # calc explained variance
        explained_variance_ = ((s ** 2) / (n_samples - 1)) / X.shape[1]
        # compute PC scores
        pc_scores = X @ Va.T
        # get loadings from eigenvectors
        loadings =  Va.T @ np.diag(s)
        loadings /= np.sqrt(X.shape[0]-1)
        return PCAResults(
            pc_scores=pc_scores,
            loadings=loadings, 
            explained_variance=explained_variance_
        )
    
    