"""
Co-activation pattern (CAP) analysis and bilinear regression for fMRI data.

"""

from dataclasses import dataclass
from typing import Literal, List

import numpy as np
import pandas as pd

from numpy.linalg import pinv
from scipy.signal import find_peaks
from scipy.stats import zscore

from sklearn.cluster import KMeans

from task_arousal.analysis.basis import create_spline_event_reg
from task_arousal.analysis.utils import get_trials_from_event_dfs
from task_arousal.constants import EVENT_COLUMNS, SLICE_TIMING_REF

# define the resampling of the event time course for boxcar function (in seconds)
RESAMPLE_TR = 0.01  # seconds


@dataclass
class CAPResults:
    cap_maps: np.ndarray  # Shape: (n_caps, n_voxels)
    cap_occurrences: np.ndarray  # Shape: (n_timepoints,)
    peak_indices: np.ndarray  # Shape: (n_peaks,)
    peak_threshold: float
    global_signal: np.ndarray  # Shape: (n_timepoints,)
    normalized: bool


class CAP:
    """Co-activation pattern (CAP) analysis on (n_timepoints, n_voxels) matrices.

    CAP identifies recurring, high-amplitude whole-brain patterns by:
    1. Extracting time points where a global signal exceeds a threshold.
    2. Extracting whole-brain volumes at those time points.
    3. Clustering extracted volumes into CAPs using k-means clustering.
    4. Returning CAP maps and their temporal occurrences.

    Notes
    -----
    - This module assumes fMRI data are provided as a 2D array with shape
        (n_timepoints, n_voxels).
    - Peak detection is the critical step in CAP; the implementation is separated
        into `detect_global_signal_peaks` and controlled by `CAP` init parameters.

    """

    def __init__(
        self,
        n_caps: int = 6,
        global_signal_reducer: Literal["mean", "median"] = "mean",
        gray_matter_mask: np.ndarray | None = None,
        peak_method: Literal["zscore", "percentile", "absolute"] = "zscore",
        peak_threshold: float = 2.0,
        peak_min_distance: int = 0,
        peak_max_peaks: int | None = None,
        peak_use_local_maxima: bool = True,
        peak_prominence: float | tuple[float, float] | None = None,
        peak_width: float | tuple[float, float] | None = None,
        peak_wlen: int | None = None,
        peak_rel_height: float = 0.5,
        peak_plateau_size: int | tuple[int, int] | None = None,
        standardize_volumes: bool = False,
        kmeans_n_init: int | Literal["auto"] = "auto",
        kmeans_max_iter: int = 300,
        random_state: int | None = 0,
        eps: float = 1e-12,
    ):
        self.n_caps = int(n_caps)
        self.global_signal_reducer = global_signal_reducer
        self.gray_matter_mask = gray_matter_mask
        self.peak_method: Literal["zscore", "percentile", "absolute"] = peak_method
        self.peak_threshold = float(peak_threshold)
        self.peak_min_distance = int(peak_min_distance)
        self.peak_max_peaks = peak_max_peaks
        self.peak_use_local_maxima = bool(peak_use_local_maxima)
        self.peak_prominence = peak_prominence
        self.peak_width = peak_width
        self.peak_wlen = peak_wlen
        self.peak_rel_height = float(peak_rel_height)
        self.peak_plateau_size = peak_plateau_size
        if standardize_volumes:
            print(
                "standardizing cap volumes before clustering - Note: may affect interpretability"
                " especially if passing the cap maps to bilinear regression!"
            )
        self.standardize_volumes = bool(standardize_volumes)
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.random_state = random_state
        self.eps = float(eps)

        if self.n_caps <= 0:
            raise ValueError("n_caps must be > 0")

    def _compute_global_signal(self, X: np.ndarray) -> np.ndarray:
        if self.gray_matter_mask is not None:
            gm = np.asarray(self.gray_matter_mask).astype(bool)
            if gm.ndim != 1:
                raise ValueError(f"gray_matter_mask must be 1D, got shape {gm.shape}")
            if gm.shape[0] != X.shape[1]:
                raise ValueError(
                    f"gray_matter_mask length ({gm.shape[0]}) must match n_voxels ({X.shape[1]})"
                )
            if int(np.sum(gm)) == 0:
                raise ValueError("gray_matter_mask contains no True voxels")
            X = X[:, gm]

        if self.global_signal_reducer == "mean":
            return np.mean(X, axis=1)
        if self.global_signal_reducer == "median":
            return np.median(X, axis=1)
        raise ValueError(
            f"Unknown global_signal_reducer={self.global_signal_reducer!r}"
        )

    def _standardize_rows(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        mu = A.mean(axis=1, keepdims=True)
        sd = A.std(axis=1, keepdims=True)
        return (A - mu) / (sd + self.eps)

    def detect(self, X: np.ndarray) -> CAPResults:
        """Run CAP analysis.

        Parameters
        ----------
        X : np.ndarray
            fMRI data matrix of shape (n_timepoints, n_voxels).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_timepoints, n_voxels), got shape {X.shape}"
            )

        n_timepoints, _ = X.shape
        global_signal = self._compute_global_signal(X)

        peak_indices, used_threshold = detect_global_signal_peaks(
            global_signal,
            method=self.peak_method,
            threshold=self.peak_threshold,
            min_distance=self.peak_min_distance,
            max_peaks=self.peak_max_peaks,
            use_local_maxima=self.peak_use_local_maxima,
            prominence=self.peak_prominence,
            width=self.peak_width,
            wlen=self.peak_wlen,
            rel_height=self.peak_rel_height,
            plateau_size=self.peak_plateau_size,
        )

        if peak_indices.size == 0:
            raise ValueError(
                "No peaks detected. Consider lowering peak_threshold, switching peak_method, "
                "or disabling local-maxima peak requirement."
            )
        if peak_indices.size < self.n_caps:
            raise ValueError(
                f"Detected {peak_indices.size} peaks but n_caps={self.n_caps}. "
                "Decrease n_caps or detect more peaks."
            )

        peak_volumes = X[peak_indices, :]
        if self.standardize_volumes:
            peak_volumes = self._standardize_rows(peak_volumes)

        km = KMeans(
            n_clusters=self.n_caps,
            n_init=self.kmeans_n_init
            if isinstance(self.kmeans_n_init, int)
            else "auto",
            max_iter=self.kmeans_max_iter,
            random_state=self.random_state,
        )
        labels = km.fit_predict(peak_volumes)

        cap_occurrences = np.full((n_timepoints,), fill_value=-1, dtype=int)
        cap_occurrences[peak_indices] = labels

        cap_maps = np.asarray(km.cluster_centers_, dtype=float)

        return CAPResults(
            cap_maps=cap_maps,
            cap_occurrences=cap_occurrences,
            peak_indices=peak_indices,
            peak_threshold=used_threshold,
            global_signal=global_signal,
            normalized=self.standardize_volumes,
        )


class BilinearFMRI:
    """
    Bilinear (tensor) regression for fMRI using spatial CAPs and temporal spline bases (see
    dlm.py for spline basis construction documentation).

    Model:
        Y ~= (X A^T) S

    Where:
        Y : (T, V)   fMRI data
        S : (K, V)   CAP spatial patterns
        X : (T, B)   spline regressors
        A : (K, B)   coupling coefficients (estimated)

    Closed-form (ridge) solution (when S and X are fixed):
        (S S^T + lambda I_K) A (X^T X + lambda I_B) = S Y^T X

    This module:
        - Fits the bilinear least-squares (or ridge) solution
        - Allows reconstruction back into voxel and time space
    """

    def __init__(
        self,
        tr: float,
        normalize_S: Literal["l2", "maxabs"] | None = "l2",
        ridge_lambda: float = 1e-2,
        regressor_extend: float = 10.0,
        knots_per_sec: float = 0.3,
        n_knots: int | None = None,
        knots: List[int] | None = None,
        basis_type: Literal["cr", "bs"] = "cr",
        regressor_duration: float | None = None,
        center: bool = True,
        eps: float = 1e-12,
    ):
        """
        Parameters
        ----------
        tr : float
            Repetition time (TR) of fMRI data in seconds.
        normalize_S : {'l2', 'maxabs'} | None
            Optional normalization of CAP maps to fix scale indeterminacy and ensure comparability of coupling coefficients across CAPs.
        ridge_lambda : float
            Ridge regularization parameter (0 = ordinary least squares). Default 1e-2.
        regressor_extend: float
            how much time (in seconds) after the end of the event to extend the regressor. If None, the regressor
            will only cover the duration of the event. Defaults is 10 seconds. If regressor_duration is set, this parameter is ignored.
        knots_per_sec: float
            number of knots per second in the spline basis across temporal lags. This ensures
            that varying duration trials have similar temporal smoothness in the basis. For example,
            a value of 0.5 results in one knot every 2 seconds. Default is 0.5 knots per second.
        n_knots: int | None
            fix the number of knots in the spline basis across temporal lags. If this parameter is set,
            the knots_per_sec parameter is ignored. Default is None.
        knots: List[int] | None
            knot locations for the spline basis across temporal lags. If supplied, this
            overrides the n_knots parameter. If this parameter is set, the knots_per_sec parameter and
            n_knots parameter are ignored.
        basis_type: Literal['cr','bs']
            basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.
        regressor_duration: float | None
            fix the duration of all spline regressors - i.e. the duration after onset of the event.
            If set to None, the regressor duration will be set to the event duration from the event data.
            Note, that if regressor_duration is None, the number of lags (nlags) will vary across events.
        center : bool
            Whether to mean-center Y and X before fitting.
        eps : float
            Small constant to avoid division by zero.
        """
        self.tr = tr
        self.lam = ridge_lambda
        self.normalize_S = normalize_S
        self.regressor_extend = regressor_extend
        self.knots_per_sec = knots_per_sec
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis_type
        self.regressor_duration = regressor_duration
        self.center = center
        self.eps = eps

        # save S for reconstruction
        self.S = None  # (K, V)
        # Learned quantities
        self.A = None  # (K, B)
        self.Z = None  # (T, K) CAP expression timecourses
        self.Y_hat = None  # (T, V) reconstructed signal

        # Centering metadata
        self._center = False
        self.Y_mean_ = None  # (1, V)
        self.X_mean_ = None  # (1, B)

    # -----------------------------------------------------

    def fit(
        self,
        event_dfs: List[pd.DataFrame],
        fmri_data: List[np.ndarray],
        cap_maps: np.ndarray,
        compute_Y_hat: bool = True,
    ):
        """
        Fit bilinear regression model.

        Parameters
        ----------
        event_dfs: list of pd.DataFrame
            List of event dataframes for each run. Used to construct X (T, B) where
            T is number of temporally concatenated time points and B is number of basis functions.
        fmri_data: list of np.ndarray
            List of fMRI data arrays (time points x voxels) for each run. Used to construct Y (T, V) where
            T is the temporally concatenated time points across runs and V is number of voxels.
        cap_maps: np.ndarray
            CAP spatial patterns - X (K, V) where K is number of CAPs and V is number of voxels.

        Y : array, shape (T, V)
            Task fMRI data
        S : array, shape (K, V)
            CAP spatial patterns
        X : array, shape (T, B)
            Temporal spline regressors

        Returns
        -------
        self
        """

        # check that event_dfs and outcome_data have same length
        if len(event_dfs) != len(fmri_data):
            raise ValueError("event_dfs and fmri_data must have the same length")
        # check that event_dfs have required columns
        for i, df in enumerate(event_dfs):
            if not all(col in df.columns for col in EVENT_COLUMNS):
                raise ValueError(f"Missing columns: {EVENT_COLUMNS} in dataframe {i}")

        # get trial types from all event dfs
        self.trial_types = get_trials_from_event_dfs(event_dfs)

        # create event regressors for each session/run
        (
            self.event_regs,
            self.nlags,
            self.basis,
            self.trial_durations,
            self.trial_durations_extend,
        ) = create_spline_event_reg(
            event_dfs=event_dfs,
            outcome_data=fmri_data,
            trial_types=self.trial_types,
            tr=self.tr,
            resample_tr=RESAMPLE_TR,
            slice_timing_ref=SLICE_TIMING_REF,
            knots_per_sec=self.knots_per_sec,
            n_knots=self.n_knots,
            knots=self.knots,
            basis_type=self.basis_type,
            regressor_duration=self.regressor_duration,
            regressor_extend=self.regressor_extend,
        )

        # create column names for event regressors
        self.event_reg_cols = [
            f"{trial}_lag_spline{n + 1}"
            for trial in self.trial_types
            for n in range(self.basis[trial]._n_knots)
        ]

        # concatenate and z-score temporal regressors across runs
        X = np.vstack(self.event_regs)
        X = np.array(zscore(X, axis=0))

        # concatenate and z-score fMRI data across runs
        Y = np.vstack(fmri_data)
        Y = np.array(zscore(Y, axis=0))
        # free memory
        del fmri_data

        # initialize matrices
        Y = np.asarray(Y, dtype=float)
        S = np.asarray(cap_maps, dtype=float)
        X = np.asarray(X, dtype=float)

        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D (T, V), got shape {Y.shape}")
        if S.ndim != 2:
            raise ValueError(f"S must be 2D (K, V), got shape {S.shape}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (T, B), got shape {X.shape}")

        T, V = Y.shape
        K, V_s = S.shape
        T_x, B = X.shape

        if V != V_s:
            raise ValueError(f"Voxel dimension mismatch: Y has V={V}, S has V={V_s}")
        if T != T_x:
            raise ValueError(f"Time dimension mismatch: Y has T={T}, X has T={T_x}")

        # --------
        # Optional CAP-map normalization (for coefficient comparability)
        # --------
        self.S_scale_ = None
        if self.normalize_S is not None:
            if self.normalize_S == "l2":
                scale = np.linalg.norm(S, axis=1, keepdims=True)
            elif self.normalize_S == "maxabs":
                scale = np.max(np.abs(S), axis=1, keepdims=True)
            else:
                raise ValueError(f"Unknown normalize_S={self.normalize_S!r}")

            scale = np.maximum(scale, float(self.eps))
            self.S_scale_ = scale
            S = S / scale

        self.S = S
        # --------
        # Optional centering
        # --------
        self._center = bool(self.center)
        self.Y_mean_ = None
        self.X_mean_ = None
        if self._center:
            self.Y_mean_ = Y.mean(axis=0, keepdims=True)
            self.X_mean_ = X.mean(axis=0, keepdims=True)
            Y = Y - self.Y_mean_
            X = X - self.X_mean_

        # --------
        # STEP 1: spatial least squares
        #
        # Solve: Y ≈ Z @ S   (Z: T x K)
        # Z = Y S^T (S S^T)^-1
        # --------
        SS = self.S @ self.S.T  # (K, K)
        YSt = Y @ self.S.T  # (T, K)

        if self.lam > 0:
            SS_reg = SS + float(self.lam) * np.eye(K)
            Z = np.linalg.solve(SS_reg, YSt.T).T
        else:
            Z = YSt @ pinv(SS)

        # --------
        # STEP 2: temporal least squares
        #
        # Solve: Z ≈ X @ A.T   (A: K x B)
        # --------
        XtX = X.T @ X  # (B, B)
        ZtX = Z.T @ X  # (K, B)

        if self.lam > 0:
            XtX_reg = XtX + float(self.lam) * np.eye(B)
            A = np.linalg.solve(XtX_reg, ZtX.T).T
        else:
            A = ZtX @ pinv(XtX)

        # --------
        # Save fitted quantities
        # --------
        self.A = A
        self.Z = Z
        if compute_Y_hat:
            Y_hat = Z @ self.S
            if self._center and self.Y_mean_ is not None:
                Y_hat = Y_hat + self.Y_mean_
            self.Y_hat = Y_hat
        else:
            self.Y_hat = None

        return self

    def project_trial_coefficents(self, trial: str) -> np.ndarray:
        """
        Project trial-specific coefficients of temporal spline regressors (task structure) in
        bilinear coupling matrix A back into time space (time since trial onset).

        Parameters
        ----------
        trial : str
            Name of the trial to project.

        """
        if self.A is None:
            raise ValueError("Model not fit")

        # get number of knots from basis
        trial_n_knots = self.basis[trial]._n_knots
        # get indices of trial from event column labels
        trial_idx = [
            self.event_reg_cols.index(f"{trial}_lag_spline{i + 1}")
            for i in range(trial_n_knots)
        ]

        # select trial coefficients
        A_trial = self.A[:, trial_idx]  # (K, B_trial)

        # get trial basis
        basis = self.basis[trial].basis.T  # (B_trial, L)

        # project coefficients into time space
        A_project = A_trial @ basis  # (K, L)

        return A_project

    def reconstruct_from_splines(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct voxelwise signal from spline regressors.

        Useful for projecting coefficients back into time space.

        Parameters
        ----------
        X : array, shape (T, B)

        Returns
        -------
        Y_recon : array, shape (T, V)
        """
        if self.A is None or self.S is None:
            raise ValueError("Model not fit")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (T, B), got shape {X.shape}")

        if self._center and self.X_mean_ is not None:
            X = X - self.X_mean_

        Z_hat = X @ self.A.T  # (T, K)
        Y_recon = Z_hat @ self.S  # (T, V)
        if self._center and self.Y_mean_ is not None:
            Y_recon = Y_recon + self.Y_mean_

        return Y_recon

    @property
    def cap_timecourses(self):
        """
        Return CAP expression timecourses.

        Returns
        -------
        Z : array, shape (T, K)
        """
        if self.Z is None:
            raise ValueError("Model not fit")
        return self.Z

    @property
    def coefficients(self):
        """
        Bilinear coupling matrix A.

        Returns
        -------
        A : array, shape (K, B)
        """
        if self.A is None:
            raise ValueError("Model not fit")
        return self.A


def plot_cap_results(
    results: CAPResults,
    ax=None,
    linewidth: float = 1.5,
    peak_linewidth: float = 2.0,
    peak_alpha: float = 0.9,
    cmap: str = "tab10",
    title: str | None = None,
):
    """Plot CAP results as colored peak markers on the global signal.

    This is a standalone utility and is not called by the `CAP` class.

    Parameters
    ----------
    results : CAPResults
        Output of `CAP.detect`.
    ax : matplotlib.axes.Axes | None
        Optional axes to plot into. If None, a new figure+axes is created.
    linewidth : float
        Line width for the global signal trace.
    peak_linewidth : float
        Line width for vertical CAP peak markers.
    peak_alpha : float
        Alpha for vertical CAP peak markers.
    cmap : str
        Matplotlib colormap name used to color CAP labels.
    title : str | None
        Optional plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    gs = np.asarray(results.global_signal, dtype=float)
    if gs.ndim != 1:
        raise ValueError(f"results.global_signal must be 1D, got shape {gs.shape}")

    peak_indices = np.asarray(results.peak_indices, dtype=int)
    if peak_indices.ndim != 1:
        raise ValueError(
            f"results.peak_indices must be 1D, got shape {peak_indices.shape}"
        )
    if peak_indices.size > 0 and (
        np.min(peak_indices) < 0 or np.max(peak_indices) >= gs.shape[0]
    ):
        raise ValueError("results.peak_indices out of bounds for results.global_signal")

    occ = np.asarray(results.cap_occurrences, dtype=int)
    if occ.ndim != 1:
        raise ValueError(f"results.cap_occurrences must be 1D, got shape {occ.shape}")
    if occ.shape[0] != gs.shape[0]:
        raise ValueError(
            f"results.cap_occurrences length ({occ.shape[0]}) must match global_signal length ({gs.shape[0]})"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(gs, color="black", linewidth=linewidth, label="global signal")

    if peak_indices.size > 0:
        labels_at_peaks = occ[peak_indices]
        valid = labels_at_peaks >= 0
        peak_indices_valid = peak_indices[valid]
        labels_valid = labels_at_peaks[valid]

        if peak_indices_valid.size > 0:
            unique_labels = np.unique(labels_valid)
            cm = plt.get_cmap(cmap)

            for cap_label in unique_labels:
                idx = peak_indices_valid[labels_valid == cap_label]
                color = cm(int(cap_label) % cm.N)
                ax.vlines(
                    idx,
                    ymin=np.min(gs),
                    ymax=np.max(gs),
                    colors=[color],
                    linewidth=peak_linewidth,
                    alpha=peak_alpha,
                    label=f"CAP {int(cap_label)}",
                )

    ax.set_xlabel("time (samples)")
    ax.set_ylabel("global signal")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    plt.show()

    return fig, ax


def detect_global_signal_peaks(
    global_signal: np.ndarray,
    method: Literal["zscore", "percentile", "absolute"] = "zscore",
    threshold: float = 2.0,
    min_distance: int = 0,
    max_peaks: int | None = None,
    use_local_maxima: bool = True,
    prominence: float | tuple[float, float] | None = None,
    width: float | tuple[float, float] | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Detect high-amplitude events in a 1D global signal.

    Parameters
    ----------
    global_signal : np.ndarray
        1D array of length n_timepoints.
    method : {'zscore', 'percentile', 'absolute'}
        Thresholding method.
    threshold : float
        Threshold parameter interpreted per method:
        - zscore: z-threshold applied to z-scored signal
        - percentile: percentile in [0, 100] applied to raw signal
        - absolute: amplitude threshold applied to raw signal
    min_distance : int
        Minimum separation (in samples) between peaks.
    max_peaks : int | None
        Optional cap on number of peaks kept (highest amplitude first).
    use_local_maxima : bool
        If True, detect peaks with `scipy.signal.find_peaks` (strict local maxima).
        If False, return all samples above threshold (no strict peak constraint).
    prominence, width, wlen, rel_height, plateau_size
        Passed through to `scipy.signal.find_peaks`.

    Returns
    -------
    peak_indices, used_threshold
    """
    x = np.asarray(global_signal, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"global_signal must be 1D, got shape {x.shape}")

    if method == "zscore":
        x_thr = np.array(zscore(x))
        used_threshold = float(threshold)
        mask = x_thr >= used_threshold
        score = x_thr
    elif method == "percentile":
        used_threshold = float(np.percentile(x, threshold))
        mask = x >= used_threshold
        score = x
    elif method == "absolute":
        used_threshold = float(threshold)
        mask = x >= used_threshold
        score = x
    else:
        raise ValueError(f"Unknown method={method!r}")

    if not use_local_maxima:
        candidates = np.where(mask)[0].astype(int)
        return candidates, used_threshold

    peaks, props = find_peaks(
        score,
        height=used_threshold,
        distance=int(min_distance) if int(min_distance) > 0 else None,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )
    candidates = np.asarray(peaks, dtype=int)

    if candidates.size == 0:
        return candidates, used_threshold

    peak_heights = props.get("peak_heights")
    if peak_heights is None:
        peak_heights = score[candidates]
    peak_heights = np.asarray(peak_heights, dtype=float)

    if max_peaks is not None and candidates.size > int(max_peaks):
        order = np.argsort(peak_heights)[::-1]
        candidates = np.array(sorted(candidates[order[: int(max_peaks)]]), dtype=int)

    return candidates, used_threshold


# ---------------------------------------------------------
# Bilinear Regression Utilities
# ---------------------------------------------------------


def center_columns(X):
    """Mean-center columns of a matrix."""
    return X - X.mean(axis=0, keepdims=True)


def ridge_inverse(M, lam):
    """Compute (M + lam*I)^-1 safely."""
    M = np.asarray(M, dtype=float)
    lam = float(lam)
    return np.linalg.solve(M + lam * np.eye(M.shape[0]), np.eye(M.shape[0]))
