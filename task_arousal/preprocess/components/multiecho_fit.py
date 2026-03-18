"""
Module for estimation of T2* and S0 from multi-echo fMRI data using a log-linear fit. Estimation
peformed using tedana.
"""

import subprocess
import tempfile

from typing import List

import nibabel as nib
import numpy as np

from nilearn.masking import apply_mask, unmask
from scipy.sparse import csc_matrix, eye, kron
from scipy.sparse.linalg import splu
from tedana.utils import make_adaptive_mask


def fit_multiecho(fp_echos: List[str], echo_times: List[float], mask_fp: str):
    """
    Estimate T2* and S0 from multi-echo fMRI data using a log-linear fit with tedana.

    Parameters
    ----------
    fp_echos : list of str
        List of file paths to the multi-echo fMRI data, ordered by echo time.
    echo_times : list of float
        List of echo times corresponding to the multi-echo fMRI data, in milliseconds.
    mask_fp : str
        File path to the fMRIPrep functional brain mask to use for estimation.

    Returns
    -------
    t2_img : nib.Nifti1Image
        Estimated T2* image.
    s0_img : nib.Nifti1Image
        Estimated S0 image.

    """
    # input checks
    if len(fp_echos) != len(echo_times):
        raise ValueError(
            "The number of echo files must match the number of echo times."
        )
    echo_times_ms = np.asarray(echo_times, dtype=float)
    if np.any(~np.isfinite(echo_times_ms)) or np.any(echo_times_ms <= 0):
        raise ValueError("Echo times must be finite, positive values in milliseconds.")
    if np.max(echo_times_ms) < 1:
        raise ValueError(
            "Echo times must be provided in milliseconds. Values smaller than 1 suggest seconds."
        )
    # load mask image
    mask_img = nib.nifti1.load(mask_fp)
    assert isinstance(mask_img, nib.nifti1.Nifti1Image), (
        f"Could not load mask image from {mask_fp}"
    )
    # load multi-echo data
    catd = _load_echo_data(fp_echos, mask_img)
    # compute t2* and S0 images from multiecho data
    # get adaptive mask
    mask_denoise, masksum_denoise = make_adaptive_mask(
        catd,
        threshold=2,  # must have at least 2 echoes with good signal to be included in the fit
        methods=["dropout"],  # default method
    )
    # estimate t2* and S0 images
    td_estimator = TemporalDecayEstimator(
        TE=echo_times_ms,
        T=catd.shape[2],
        lambda0=3.0,
        lambda1=1.0,
        min_signal=1e-6,
        te_rescale_factor=10.0,
    )
    s0_full_ts, t2s_full_ts = td_estimator.fit(
        data=catd,
        adaptive_mask=masksum_denoise,
    )

    # unmask t2* and S0 time series to original 4D shape
    t2s_full_ts_img = unmask(t2s_full_ts.T, mask_img)
    s0_full_ts_img = unmask(s0_full_ts.T, mask_img)
    return t2s_full_ts_img, s0_full_ts_img


def multiecho_to_std(
    img: str | nib.nifti1.Nifti1Image,
    std_space_ref_fp: str,
    native_to_t1w_fp: str,
    t1w_to_std_fp: str,
    output_fp: str | None = None,
) -> nib.nifti1.Nifti1Image:
    """
    Apply the standard fMRIPrep spatial transformations to the given image, to bring it into MNI space. This is necessary
    for the T2* and S0 images estimated from multi-echo data, which are in the same space as the original BOLD data and thus require
    the same transformations to be applied.

    Inspired from:
    https://tedana.readthedocs.io/en/stable/faq.html#warping-scanner-space-fmriprep-outputs-to-standard-space

    Parameters
    ----------
    img : str or nib.Nifti1Image
        Image to transform. If a NIfTI image is provided, it will be written to a
        temporary directory before calling ANTs.
    std_space_ref_fp : str
        File path to the standard space reference image (e.g. MNI152NLin2009cAsym).
    native_to_t1w_fp : str
        File path to the fMRIPrep-generated transformation file from native space to T1w space (e.g. from-boldref_to-T1w_mode-image_xfm.txt).
    t1w_to_std_fp : str
        File path to the fMRIPrep-generated transformation file from T1w space to standard space (e.g. from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5).
    output_fp : str or None
        File path where the transformed image should be saved. If None, a
        temporary output path is used and the transformed image is returned in memory.

    Returns
    -------
    nib.Nifti1Image
        The transformed image.

    """

    def _load_materialized_nifti(img_fp: str) -> nib.nifti1.Nifti1Image:
        loaded_img = nib.nifti1.load(img_fp)
        if not isinstance(loaded_img, nib.nifti1.Nifti1Image):
            raise TypeError(f"Expected NIfTI image at {img_fp}, got {type(loaded_img)}")
        return nib.nifti1.Nifti1Image(
            np.asanyarray(loaded_img.dataobj),
            affine=loaded_img.affine,
            header=loaded_img.header.copy(),
            extra=loaded_img.extra.copy(),
        )

    with tempfile.TemporaryDirectory(prefix="task_arousal_multiecho_") as tmpdir:
        if isinstance(img, nib.nifti1.Nifti1Image):
            img_fp = f"{tmpdir}/multiecho_native.nii.gz"
            nib.nifti1.save(img, img_fp)
        else:
            img_fp = img

        resolved_output_fp = output_fp or f"{tmpdir}/multiecho_std.nii.gz"

        try:
            subprocess.run(
                [
                    "antsApplyTransforms",
                    "-e",
                    "3",
                    "-i",
                    img_fp,
                    "-r",
                    std_space_ref_fp,
                    "-o",
                    resolved_output_fp,
                    "-n",
                    "LanczosWindowedSinc",
                    "-t",
                    t1w_to_std_fp,
                    "-t",
                    native_to_t1w_fp,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "antsApplyTransforms failed while transforming multi-echo image: "
                f"{exc.stderr.strip() or exc.stdout.strip() or exc}"
            ) from exc

        if output_fp is None:
            return _load_materialized_nifti(resolved_output_fp)

    saved_img = nib.nifti1.load(output_fp)
    if not isinstance(saved_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Expected NIfTI image at {output_fp}, got {type(saved_img)}")
    return saved_img


class TemporalDecayEstimator:
    r"""Estimate time-varying S0 and T2* with quadratic temporal smoothing.

    This class fits a monoexponential decay model to the multi-echo signal at each
    voxel while coupling neighboring timepoints through a second-difference penalty.
    For a voxel with signal :math:`S(TE, t)` at echo time ``TE`` and timepoint ``t``,
    the fitted model is

    .. math::

        \log S(TE, t) = \beta_0(t) - \beta_1(t) \, TE,

    where :math:`\beta_0(t) = \log S_0(t)` and :math:`\beta_1(t) = 1 / T_2^*(t)`.
    The unknown parameter vector is ordered in an interleaved form,

    .. math::

        [\beta_0(t_0), \beta_1(t_0), \beta_0(t_1), \beta_1(t_1), \ldots],

    so each timepoint contributes a local intercept and decay-rate pair.

    The estimator solves the penalized least-squares problem

    .. math::

        \min_{\beta}\; \|y - Z\beta\|^2 + \beta^T P\beta,

    where ``y`` is the stacked log-signal over echoes and time, ``Z`` is the block
    design matrix built from ``[1, -TE]`` at each timepoint, and ``P`` is a quadratic
    smoothness penalty constructed from the temporal second-difference operator.
    The smoothing terms encourage :math:`\log S_0(t)` and the decay rate
    :math:`1 / T_2^*(t)` to change smoothly over time without forcing them to be
    constant.

    In practice, the full design matrix is never materialized. Instead, the class
    precomputes a sparse solver for each possible number of usable echoes indicated by
    the adaptive mask. Voxels are then grouped by their good-echo count and solved in
    batches, which preserves the full fit mathematically while avoiding repeated matrix
    construction.

    The class also supports an internal rescaling of echo times before building the
    design matrix. This changes only the numerical conditioning of the slope fit. The
    fitted decay-rate parameter is converted back so the reported T2* values always
    remain in milliseconds.

    Notes
    -----
    - Echo times are expected in milliseconds.
    - The fit is performed in log space and therefore requires strictly positive signal
      for the echoes included in the fit.
        - ``te_rescale_factor`` divides the echo times used internally by the solver. This
            does not change the physical interpretation of the output because T2* is scaled
            back to milliseconds before being returned.
        - ``max_t2star_ms`` caps implausibly large T2* estimates that arise when the fitted
            decay rate is extremely close to zero but still positive.
    - The solver does not impose a hard positivity constraint on the decay rate during
      optimization. Non-positive fitted decay rates are returned as NaN T2* values.
    """

    def __init__(
        self,
        TE,
        T,
        lambda0=1.0,
        lambda1=1.0,
        min_signal=1e-6,
        te_rescale_factor=100.0,
        max_t2star_ms=500.0,
    ):
        """
        Parameters
        ----------
        TE : array (E,)
            Echo times in milliseconds.
        T : int
            Number of timepoints
        lambda0, lambda1 : float
            Smoothness penalties for log(S0) and decay rate.
        min_signal : float
            Minimum strictly positive signal retained before applying the log transform.
        te_rescale_factor : float
            Factor used to divide echo times internally before fitting. This can improve
            conditioning for the decay-rate parameter. Returned T2* estimates are always
            converted back to milliseconds.
        max_t2star_ms : float
            Upper bound applied to returned T2* estimates in milliseconds.
        """
        self.TE_ms = np.asarray(TE, dtype=float)
        self.E = len(self.TE_ms)
        self.T = int(T)
        self.lambda0 = float(lambda0)
        self.lambda1 = float(lambda1)
        self.min_signal = float(min_signal)
        self.te_rescale_factor = float(te_rescale_factor)
        self.max_t2star_ms = float(max_t2star_ms)

        if self.TE_ms.ndim != 1:
            raise ValueError("TE must be a one-dimensional array of echo times.")
        if self.E < 2:
            raise ValueError("At least two echo times are required.")
        if np.any(~np.isfinite(self.TE_ms)) or np.any(self.TE_ms <= 0):
            raise ValueError(
                "TE must contain finite, positive echo times in milliseconds."
            )
        if np.max(self.TE_ms) < 1:
            raise ValueError(
                "Echo times must be provided in milliseconds. Values smaller than 1 suggest seconds."
            )
        if np.any(np.diff(self.TE_ms) <= 0):
            raise ValueError("TE must be strictly increasing.")
        if self.T < 1:
            raise ValueError("T must be a positive integer.")
        if self.lambda0 < 0 or self.lambda1 < 0:
            raise ValueError("Smoothing penalties must be non-negative.")
        if self.min_signal <= 0:
            raise ValueError("min_signal must be strictly positive.")
        if self.te_rescale_factor <= 0:
            raise ValueError("te_rescale_factor must be strictly positive.")
        if self.max_t2star_ms <= 0:
            raise ValueError("max_t2star_ms must be strictly positive.")

        self.TE = self.TE_ms / self.te_rescale_factor

        self._precompute()

    # -----------------------------
    # Core precomputation
    # -----------------------------
    def _precompute(self):
        """Precompute sparse linear systems for each adaptive-mask echo count.

        The same temporal penalty is shared across voxels, but the per-timepoint echo
        design changes when a voxel has fewer usable echoes. This method builds one
        sparse system per possible good-echo count so later voxel fits only need to
        assemble the right-hand side and call the cached solver.
        """
        # Build second-difference matrix
        D = self._second_diff_matrix(self.T)
        DtD = D.T @ D  # (T, T)
        penalty = kron(
            csc_matrix(DtD),
            csc_matrix(np.diag([self.lambda0, self.lambda1])),
            format="csc",
        )

        self._design_matrices = {}
        self._solvers = {}

        for n_good_echos in range(2, self.E + 1):
            # Interleaved coefficient order is
            # [beta0(t0), beta1(t0), beta0(t1), beta1(t1), ...].
            # The per-timepoint model uses negative rescaled TE so beta1 is the decay
            # rate in the rescaled TE units, which should be positive for a physically
            # plausible monoexponential decay.
            X = np.column_stack(
                [np.ones(n_good_echos, dtype=float), -self.TE[:n_good_echos]]
            )
            XtX = X.T @ X
            A_data = kron(eye(self.T, format="csc"), csc_matrix(XtX), format="csc")
            self._design_matrices[n_good_echos] = X
            self._solvers[n_good_echos] = splu(A_data + penalty)

    # -----------------------------
    def _second_diff_matrix(self, T):
        """Construct the temporal second-difference operator.

        Each row encodes ``[1, -2, 1]`` across three consecutive timepoints. Applying
        this operator to a parameter time course measures local curvature, so squaring
        and summing these values penalizes rapid bending over time.
        """
        if T < 3:
            return np.zeros((0, T), dtype=float)
        D = np.zeros((T - 2, T))
        for t in range(T - 2):
            D[t, t : t + 3] = [1, -2, 1]
        return D

    def _validate_good_echo_count(self, n_good_echos: int) -> int:
        """Validate the adaptive-mask echo count for a voxel or voxel group."""
        if n_good_echos < 2:
            raise ValueError(
                "At least two good echoes are required to estimate T2* and S0."
            )
        if n_good_echos > self.E:
            raise ValueError(
                f"adaptive_mask entry {n_good_echos} exceeds the available echo count {self.E}."
            )
        return n_good_echos

    def _prepare_log_data(self, data: np.ndarray, n_good_echos: int) -> np.ndarray:
        """Select usable echoes and transform the signal to log space.

        The linearized model is defined on ``log(signal)``. This helper applies the
        adaptive-mask echo cutoff, checks that the retained data are finite, floors
        very small positive values to ``min_signal``, and returns the log-transformed
        time-by-echo array for a single voxel.
        """
        signal = np.asarray(data[:, :n_good_echos], dtype=float)
        if signal.shape != (self.T, n_good_echos):
            raise ValueError(
                f"Expected voxel data with shape ({self.T}, {n_good_echos}), got {signal.shape}."
            )
        if np.any(~np.isfinite(signal)):
            raise ValueError(
                "Selected echoes contain non-finite signal. Use the adaptive mask to "
                "exclude these voxels before fitting."
            )
        return np.log(np.clip(signal, a_min=self.min_signal, a_max=None))

    # -----------------------------
    # RHS computation
    # -----------------------------
    def _compute_rhs(self, log_signal, X):
        """Form the data term :math:`Z^T y` for one voxel.

        Parameters
        ----------
        log_signal : ndarray of shape (T, E_good)
            Log-transformed signal for one voxel.
        X : ndarray of shape (E_good, 2)
            Per-timepoint design matrix ``[1, -TE]``.

        Returns
        -------
        ndarray of shape (2T,)
            Right-hand side of the normal equations in interleaved coefficient order.
        """
        rhs_pairs = log_signal @ X
        return rhs_pairs.reshape(2 * self.T)

    def _solve_group(self, log_data: np.ndarray, n_good_echos: int):
        """Solve the penalized system for a batch of voxels with the same echo count.

        Parameters
        ----------
        log_data : ndarray of shape (V_group, T, E_good)
            Log-transformed voxel data for one adaptive-mask group.
        n_good_echos : int
            Number of echoes retained for all voxels in this group.

        Returns
        -------
        s0 : ndarray of shape (V_group, T)
            Estimated S0 time series.
        t2star : ndarray of shape (V_group, T)
            Estimated T2* time series in milliseconds. Entries with non-positive fitted
            decay rate are returned as NaN.
        """
        solver = self._solvers[n_good_echos]
        X = self._design_matrices[n_good_echos]
        xty = np.einsum("vte,ek->vtk", log_data, X)
        rhs = xty.reshape(log_data.shape[0], 2 * self.T).T
        beta = solver.solve(rhs).T.reshape(log_data.shape[0], self.T, 2)

        beta0 = beta[:, :, 0]
        beta1 = beta[:, :, 1]
        s0 = np.exp(beta0)
        t2star = np.full(beta1.shape, np.nan, dtype=float)
        np.divide(self.te_rescale_factor, beta1, out=t2star, where=beta1 > 0)
        np.clip(t2star, a_min=None, a_max=self.max_t2star_ms, out=t2star)

        return s0, t2star

    # -----------------------------
    # Fit single voxel
    # -----------------------------
    def fit_voxel(self, y, adaptive_mask: int | None = None):
        """Fit the temporally smoothed decay model for a single voxel.

        Parameters
        ----------
        y : array_like
            Voxel time series ordered as time x echo.
        adaptive_mask : int or None
            Number of good echoes for this voxel. If None, all echoes are used.
        """
        n_good_echos = self._validate_good_echo_count(adaptive_mask or self.E)
        log_signal = self._prepare_log_data(np.asarray(y), n_good_echos)
        s0, t2star = self._solve_group(log_signal[None, ...], n_good_echos)
        return s0[0], t2star[0]

    # -----------------------------
    # Fit many voxels (vectorized RHS)
    # -----------------------------
    def fit(self, data, adaptive_mask: np.ndarray | None = None):
        """Fit the temporally smoothed decay model for many voxels.

        Voxels are grouped by their adaptive-mask value, meaning the number of echoes
        deemed usable for that voxel. Each group is then solved with the matching
        precomputed sparse system.

        Parameters
        ----------
        data : ndarray of shape (V, E, T)
            Multi-echo voxel data ordered as voxel x echo x time, matching the
            layout expected by ``tedana.decay.fit_decay_ts``.
        adaptive_mask : ndarray of shape (V,), optional
            Integer array where each entry gives the number of good echoes for a voxel.
            If omitted, all echoes are used for every voxel.

        Returns:
        --------
        S0: (V, T)
            Estimated S0 time series for all voxels.
        T2star: (V, T)
            Estimated T2* time series in milliseconds for all voxels.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError("data must have shape (voxels, echo, time).")

        V, E, T = data.shape
        if T != self.T:
            raise ValueError(f"Expected {self.T} timepoints, got {T}.")
        if E != self.E:
            raise ValueError(f"Expected {self.E} echoes, got {E}.")

        if adaptive_mask is None:
            adaptive_mask_array = np.full(V, self.E, dtype=int)
        else:
            adaptive_mask_array = np.asarray(adaptive_mask, dtype=int)
            if adaptive_mask_array.shape != (V,):
                raise ValueError(
                    f"adaptive_mask must have shape ({V},), got {adaptive_mask_array.shape}."
                )

        S0_all = np.full((V, T), np.nan, dtype=float)
        T2_all = np.full((V, T), np.nan, dtype=float)

        for n_good_echos in np.unique(adaptive_mask_array):
            if n_good_echos < 2:
                continue
            n_good_echos = self._validate_good_echo_count(int(n_good_echos))
            voxel_idx = np.where(adaptive_mask_array == n_good_echos)[0]
            if voxel_idx.size == 0:
                continue

            group_data = data[voxel_idx, :n_good_echos, :]
            valid_signal = np.all(np.isfinite(group_data), axis=(1, 2))
            if not np.any(valid_signal):
                continue

            valid_voxel_idx = voxel_idx[valid_signal]
            clipped_group_data = np.clip(
                group_data[valid_signal], a_min=self.min_signal, a_max=None
            )
            log_data = np.log(np.transpose(clipped_group_data, (0, 2, 1)))
            s0, t2star = self._solve_group(log_data, n_good_echos)
            S0_all[valid_voxel_idx] = s0
            T2_all[valid_voxel_idx] = t2star

        return S0_all, T2_all


def _load_echo_data(fp_echos: List[str], mask_img: nib.nifti1.Nifti1Image):
    """
    Load multi-echo fMRI data from the given file paths. The tedana function
    that creates the adaptive mask expects the data to be in a 3D array of shape (brain voxels x echos [x time]),
    so the data is loaded, masked and concatenated across echoes into this format.

    Parameters
    ----------
    fp_echos : list of str
        List of file paths to the multi-echo fMRI data, ordered by echo time.
    mask_img : nib.nifti1.Nifti1Image
        NIfTI image of the fMRIPrep functional brain mask to use for estimation.

    Returns
    -------
    data : numpy.ndarray
        3D array of shape (brain voxels, echoes, time).
    """
    # load data into 3d array (S x E [x T]) array_like
    # where `S` is samples, `E` is echos, and `T` is time
    catd = []
    for fp_echo in fp_echos:
        echo_img = nib.nifti1.load(fp_echo)
        # apply mask and reshape to (brain voxels x time)
        echo_data_masked = apply_mask(
            echo_img, mask_img
        ).T  # shape (brain voxels x time)
        catd.append(echo_data_masked)
    catd = np.stack(catd, axis=1)  # shape (brain voxels x echos [x time])
    return catd
