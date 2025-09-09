"""
Module for performing complex-valued PCA on functional MRI data.
"""

from dataclasses import dataclass, field

import fbpca
import numpy as np

from scipy.signal import hilbert

@dataclass
class ComplexPCAResults:
    """
    Class for storing results of complex-valued PCA. 

    Attributes
    ----------
    pc_scores: np.ndarray
        the PC scores of the complex-valued PCA model

    loadings: np.ndarray
        the loadings of the complex-valued PCA model

    explained_variance: np.ndarray
        the explained variance of the complex-valued PCA model
    """

    pc_scores: np.ndarray
    loadings: np.ndarray
    explained_variance: np.ndarray
    loadings_phase: np.ndarray = field(init=False)
    loadings_amp: np.ndarray = field(init=False)

    def __post_init__(self):
        self.loadings_phase = np.angle(self.loadings)
        self.loadings_amp = np.abs(self.loadings)


@dataclass
class ComplexPCAReconResults:
    """
    Class for storing results of complex-valued PCA reconstruction of spatiotemporal patterns. 
    Provides utilities for writing reconstructed time courses to func.gii files.

    Attributes
    ----------
    bin_timepoints: np.ndarray
        the reconstructed time courses of the complex-valued PCA model

    bin_centers: np.ndarray
        the bin centers of the complex-valued PCA model
    """

    bin_values: np.ndarray
    bin_centers: np.ndarray


class ComplexPCA:
    def __init__(self, n_components: int = 10, n_iter: int = 10):
        self.n_components = n_components
        self.n_iter = n_iter

    def decompose(self, X: np.ndarray) -> ComplexPCAResults:
        # get number of observations
        n_samples = X.shape[0]
        # perform hilbert transform (overwrite X to save memory)
        X = hilbert_transform(X)
        # fbpca pca
        (U, s, Va) = fbpca.pca(X, k=self.n_components, n_iter=self.n_iter)
        # calc explained variance
        explained_variance_ = ((s ** 2) / (n_samples - 1)) / X.shape[1]
        # compute PC scores
        pc_scores = X @ Va.T
        # get loadings from eigenvectors
        loadings =  Va.T @ np.diag(s)
        loadings /= np.sqrt(X.shape[0]-1)
        # store SVD results in class
        self.U = U
        self.s = s
        self.Va = Va
        self.pc_scores = pc_scores
        return ComplexPCAResults(
            pc_scores=pc_scores, 
            loadings=loadings, 
            explained_variance=explained_variance_
        )

    def reconstruct(self, i: int, n_bins: int = 20) -> ComplexPCAReconResults:
        """
        Reconstruct spatiotemporal pattern for PC i from 
        complex-valued PCA results via averaging across timepoints within
        phase bins.

        Parameters
        ----------
        i : int
            Index of PC to reconstruct
        n_bins : int, optional
            Number of bins to divide phase time series into. Default is 10.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing spatiotemporal pattern and bin centers.
        """
        # reconstruct time series from PC i
        recon_ts = self._reconstruct_ts(i)
        # get phase time series
        phase_ts = np.angle(self.pc_scores[:,i])
        # shift phase delay angles from -pi to pi -> 0 to 2*pi
        phase_ts = np.mod(phase_ts, 2*np.pi)
        # bin time courses into phase bins
        bin_indx, bin_centers = _create_bins(phase_ts, n_bins)
        # average time courses within bins
        bin_values = _average_bins(recon_ts, bin_indx, n_bins)
        # return bin timepoints
        return ComplexPCAReconResults(
            bin_values=bin_values, 
            bin_centers=bin_centers
        )

    def _reconstruct_ts(
        self,
        i: int, 
        real: bool = True
    ) -> np.ndarray:
        """
        Reconstruct single time series for PC i from complex-valued PCA results by
        taking the real or imaginary part of the PC projection.

        Parameters
        ----------
        i : int
            Index of PC to reconstruct
        real : bool, optional
            Whether to return the real or imaginary part of the PC projection.
            Default is True.
        """
        U = self.U[:,[i]]
        s = np.atleast_2d(self.s[[i]])
        Va = self.Va[[i],:].conj()
        recon_ts = U @ s @ Va
        if real:
            recon_ts = np.real(recon_ts)
        else:
            recon_ts = np.imag(recon_ts)
        return recon_ts



def hilbert_transform(input_data: np.ndarray) -> np.ndarray:
    """
    Apply Hilbert transform to input data. Complex-valued data is returned.
    """
    result = hilbert(input_data, axis=0)
    # the conjugate of the transformed data is taken so as to ensure that 
    # the phase angles of the principal components progress in the rightward
    # direction.
    return result.conj() # type: ignore


def _average_bins(
    recon_ts: np.ndarray, 
    bin_indx: np.ndarray, 
    n_bins: int
) -> np.ndarray:
    """
    Average reconstructed time series within phase bins.
    """
    bin_timepoints = []
    for n in range(1, n_bins+1):
        ts_indx = np.where(bin_indx==n)[0]
        bin_timepoints.append(np.mean(recon_ts[ts_indx,:], axis=0))
    return np.array(bin_timepoints)


def _create_bins(phase_ts: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide phase time series into bins for subsequent averaging.
    """
    freq, bins = np.histogram(phase_ts, n_bins)
    bin_indx = np.digitize(phase_ts, bins)
    bin_centers = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
    return bin_indx, bin_centers

