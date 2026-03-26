"""Distributed lag modeling of TE-dependent echo curves on multi-echo data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import warnings

import numpy as np

from patsy import dmatrix  # type: ignore
from scipy.sparse import csc_matrix, eye, hstack, kron, vstack

from task_arousal.quadratic_penalty import (
    QuadraticPenaltySolver,
    as_csc,
    ridge_penalty,
    second_difference_penalty,
    zero_penalty,
)


@dataclass
class EchoCurvePredictionParams:
    pred_val: float
    run: int
    include_intercept: bool
    regressor: str | None = None
    lag: float | None = None


@dataclass
class EchoCurvePrediction:
    echo_times_ms: np.ndarray
    pred_curve: np.ndarray
    pred_effect: np.ndarray
    pred_nuisance: np.ndarray
    params: EchoCurvePredictionParams


@dataclass
class EchoCurveAcrossLagsPredictionParams:
    pred_val: float
    run: int
    include_intercept: bool
    regressor: str
    lag_min: float
    lag_max: float
    n_eval: int | None
    eval_delta: float | None


@dataclass
class EchoCurveAcrossLagsPrediction:
    echo_times_ms: np.ndarray
    pred_lags: np.ndarray
    pred_curve: np.ndarray
    pred_effect: np.ndarray
    pred_nuisance: np.ndarray
    params: EchoCurveAcrossLagsPredictionParams


@dataclass
class EchoTimecoursePredictionParams:
    echo_index: int
    echo_time_ms: float
    run: int
    include_nuisance: bool


@dataclass
class EchoTimecoursePrediction:
    pred_timecourse: np.ndarray
    pred_effect: np.ndarray
    pred_nuisance: np.ndarray
    params: EchoTimecoursePredictionParams


@dataclass
class EchoCurveAcrossLagsForEchoPredictionParams:
    echo_index: int
    echo_time_ms: float
    pred_val: float
    run: int
    include_intercept: bool
    regressor: str
    lag_min: float
    lag_max: float
    n_eval: int | None
    eval_delta: float | None


@dataclass
class EchoCurveAcrossLagsForEchoPrediction:
    pred_lags: np.ndarray
    pred_curve: np.ndarray
    pred_effect: np.ndarray
    pred_nuisance: np.ndarray
    params: EchoCurveAcrossLagsForEchoPredictionParams


class DistributedLagEchoModel:
    """Distributed lag model for TE-dependent responses in multi-echo fMRI data.

    The model treats the observed echo dependence as a set of echo-specific
    coefficients for each regressor-of-interest. Effects are fit in raw signal space
    with a quadratic penalty that smooths neighboring echoes while still allowing
    localized deviations at individual echoes.

    Notes
    -----
    - Input fMRI data are expected as a list of tensors with shape
      ``voxel x echo x time``.
    - Regressor design matrices are expected as a list of 2D arrays with shape
      ``time x regressors`` and must match the corresponding run lengths.
    - Optional confounds are passed as a list of 2D arrays with shape
      ``time x nuisance``. Run-specific intercepts are always included internally.

    Shared-confound assumption
    --------------------------
    When confound matrices are supplied, the model estimates a **single set of
    confound coefficients shared across all runs**. This is implemented by pooling
    the cross-products ``C_r^T C_r`` and ``C_r^T Y_r`` across runs, which is
    algebraically equivalent to fitting one global confound coefficient vector that
    minimises the residual summed over all runs simultaneously.

    This assumption is valid only when the confound columns have the same
    interpretation and comparable scale in every run. A discrete cosine transform
    (DCT) drift basis — where each column is a fixed-frequency sinusoid applied
    identically to every run — satisfies this requirement well.

    Run-specific noise regressors such as head motion parameters (translations,
    rotations), aCompCor components, or FD/DVARS traces do **not** satisfy this
    requirement. Although the physical unit is the same (millimetres, radians, etc.),
    the variance, trajectory, and BOLD coupling of motion parameters differ across
    runs. Estimating a single pooled coefficient for these regressors will produce
    biased estimates: the coefficient will be pulled toward runs with large motion
    excursions and will over-correct quiet runs while under-correcting high-motion
    ones. Motion parameters should instead be treated as run-specific columns, which
    is not currently supported — the recommended workaround is to regress them out
    of the data prior to fitting this model.

    Incremental fitting
    -------------------
    The model can be fit either in one call with ``fit(...)`` or incrementally with
    repeated calls to ``fit_partial(...)`` followed by a final call to
    ``finalize_fit()``. The incremental path is mathematically equivalent to the
    one-shot fit: each call to ``fit_partial`` adds one run's sufficient statistics
    to the pooled normal equations, and ``finalize_fit`` solves the shared penalized
    system once after all runs have been accumulated.

    Use the incremental API when runs should be loaded and processed one at a time:

    >>> model = DistributedLagEchoModel(echo_times_ms)
    >>> for run_index, (run_X, run_Y, run_C) in enumerate(zip(regressors, data, confounds)):
    ...     model.fit_partial(
    ...         regressors=run_X,
    ...         data=run_Y,
    ...         confounds=run_C,
    ...         regressor_slices=regressor_slices if run_index == 0 else None,
    ...         lag_design_infos=lag_design_infos if run_index == 0 else None,
    ...     )
    >>> model.finalize_fit()

    ``regressor_slices`` and ``lag_design_infos`` are only required on the first
    ``fit_partial`` call; later calls may omit them, but if they are provided again
    they must match the original values. All runs must have the same voxel count,
    the same number of regressor columns, and the same number of shared confound
    columns.
    """

    def __init__(
        self,
        echo_times_ms: np.ndarray | list[float],
        ridge_alpha: float = 1.0,
        echo_smooth_alpha: float = 1.0,
        nuisance_ridge_alpha: float = 0.0,
    ):
        self.echo_times_ms = np.asarray(echo_times_ms, dtype=float)
        if self.echo_times_ms.ndim != 1:
            raise ValueError("echo_times_ms must be a one-dimensional array")
        if len(self.echo_times_ms) < 2:
            raise ValueError("At least two echo times are required")
        if np.any(~np.isfinite(self.echo_times_ms)) or np.any(self.echo_times_ms <= 0):
            raise ValueError("echo_times_ms must contain finite, positive values")
        if np.any(np.diff(self.echo_times_ms) <= 0):
            raise ValueError("echo_times_ms must be strictly increasing")
        if ridge_alpha < 0 or echo_smooth_alpha < 0 or nuisance_ridge_alpha < 0:
            raise ValueError("All penalty weights must be non-negative")

        self.E = len(self.echo_times_ms)
        self.ridge_alpha = float(ridge_alpha)
        self.echo_smooth_alpha = float(echo_smooth_alpha)
        self.nuisance_ridge_alpha = float(nuisance_ridge_alpha)

    def _clear_fit_results(self) -> None:
        """Drop fitted outputs so a new fit cannot accidentally reuse stale state."""
        for attr in (
            "effect_coefs_",
            "nuisance_coefs_",
            "n_regressors_",
            "n_nuisance_",
            "n_voxels_",
            "intercept_column_indices_",
            "run_regressor_designs_",
            "run_nuisance_designs_",
            "run_nuisance_column_indices_",
            "regressor_slices_",
            "lag_design_infos_",
            "_solver",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _clear_partial_fit_state(self) -> None:
        """Reset incremental sufficient statistics and bookkeeping."""
        for attr in (
            "_partial_n_voxels",
            "_partial_n_regressors",
            "_partial_shared_confound_count",
            "_partial_run_count",
            "_partial_XtX",
            "_partial_XtC",
            "_partial_CtC",
            "_partial_XtY",
            "_partial_CtY",
            "_partial_intercept_column_indices",
            "_partial_run_regressor_designs",
            "_partial_run_nuisance_designs",
            "_partial_run_nuisance_column_indices",
            "_partial_regressor_slices",
            "_partial_lag_design_infos",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _normalize_run_confounds(
        self,
        run_confounds: np.ndarray | None,
        n_timepoints: int,
    ) -> np.ndarray | None:
        """Validate one run's nuisance matrix and normalize empty input to None."""
        if run_confounds is None:
            return None
        confounds_array = np.asarray(run_confounds, dtype=float)
        if confounds_array.size == 0:
            return None
        if confounds_array.ndim != 2:
            raise ValueError("Each confound array must be 2D (time x nuisance)")
        if confounds_array.shape[0] != n_timepoints:
            raise ValueError(
                "Each confound matrix must have the same number of rows as the corresponding run"
            )
        if np.any(~np.isfinite(confounds_array)):
            raise ValueError("Confound matrices must be finite")
        return confounds_array

    def _initialize_partial_fit_state(
        self,
        n_voxels: int,
        n_regressors: int,
        shared_confound_count: int,
        regressor_slices: Mapping[str, slice] | None = None,
        lag_design_infos: Mapping[str, Any] | None = None,
    ) -> None:
        """Allocate run-wise sufficient statistics for incremental fitting."""
        self._clear_fit_results()
        self._clear_partial_fit_state()
        self._partial_n_voxels = n_voxels
        self._partial_n_regressors = n_regressors
        self._partial_shared_confound_count = shared_confound_count
        self._partial_run_count = 0
        self._partial_XtX = np.zeros((n_regressors, n_regressors), dtype=float)
        self._partial_XtC = np.zeros((n_regressors, shared_confound_count), dtype=float)
        self._partial_CtC = np.zeros(
            (shared_confound_count, shared_confound_count), dtype=float
        )
        self._partial_XtY = np.zeros((n_regressors, self.E, n_voxels), dtype=float)
        self._partial_CtY = np.zeros(
            (shared_confound_count, self.E, n_voxels), dtype=float
        )
        self._partial_intercept_column_indices: list[int] = []
        self._partial_run_regressor_designs: list[np.ndarray] = []
        self._partial_run_nuisance_designs: list[np.ndarray] = []
        self._partial_run_nuisance_column_indices: list[np.ndarray] = []
        self._partial_regressor_slices = dict(regressor_slices or {})
        self._partial_lag_design_infos = dict(lag_design_infos or {})

    def _append_intercept_column(self) -> int:
        """Grow the nuisance accumulator by one run-specific intercept column."""
        self._partial_XtC = np.pad(self._partial_XtC, ((0, 0), (0, 1)))
        self._partial_CtC = np.pad(self._partial_CtC, ((0, 1), (0, 1)))
        self._partial_CtY = np.pad(self._partial_CtY, ((0, 1), (0, 0), (0, 0)))
        intercept_index = self._partial_shared_confound_count + self._partial_run_count
        self._partial_intercept_column_indices.append(intercept_index)
        self._partial_run_count += 1
        return intercept_index

    def _store_prediction_metadata(
        self,
        regressor_slices: Mapping[str, slice] | None,
        lag_design_infos: Mapping[str, Any] | None,
    ) -> None:
        """Keep prediction metadata consistent across incremental run additions."""
        if regressor_slices is not None:
            new_slices = dict(regressor_slices)
            if (
                self._partial_regressor_slices
                and new_slices != self._partial_regressor_slices
            ):
                raise ValueError(
                    "regressor_slices must be consistent across partial fits"
                )
            self._partial_regressor_slices = new_slices
        if lag_design_infos is not None:
            new_keys = set(lag_design_infos)
            old_keys = set(self._partial_lag_design_infos)
            if old_keys and new_keys != old_keys:
                raise ValueError(
                    "lag_design_infos must be consistent across partial fits"
                )
            if not old_keys:
                self._partial_lag_design_infos = dict(lag_design_infos)

    def _get_regressor_lag_transform(self, regressor: str) -> Any:
        if regressor not in self.lag_design_infos_:
            raise ValueError(
                f"No lag design info found for {regressor!r}. Provide lag_design_infos in fit()."
            )

        design_info = self.lag_design_infos_[regressor]
        factor_infos = getattr(design_info, "factor_infos", None)
        if not factor_infos:
            raise ValueError(f"Could not recover lag basis metadata for {regressor!r}.")

        factor_info = next(iter(factor_infos.values()))
        transforms = factor_info.state.get("transforms", {})
        if len(transforms) != 1:
            raise ValueError(
                f"Could not recover a unique lag transform for {regressor!r}."
            )
        return next(iter(transforms.values()))

    def _infer_regressor_lag_bounds(self, regressor: str) -> tuple[float, float]:
        transform = self._get_regressor_lag_transform(regressor)
        all_knots = getattr(transform, "_all_knots", None)
        if all_knots is None or len(all_knots) == 0:
            raise ValueError(
                f"Could not infer lag bounds for {regressor!r}; pass lags or lag_min/lag_max explicitly."
            )
        return float(all_knots[0]), float(all_knots[-1])

    def _build_prediction_lag_grid(
        self,
        regressor: str,
        lags: np.ndarray | list[float] | None = None,
        lag_min: float | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = 1.0,
    ) -> tuple[np.ndarray, float | None]:
        if lags is not None:
            pred_lags = np.asarray(lags, dtype=float).reshape(-1)
            if pred_lags.size == 0:
                raise ValueError("lags must contain at least one value")
            if np.any(~np.isfinite(pred_lags)):
                raise ValueError("lags must be finite")
            return pred_lags, None

        default_lag_min, default_lag_max = self._infer_regressor_lag_bounds(regressor)
        lag_min = default_lag_min if lag_min is None else float(lag_min)
        lag_max = default_lag_max if lag_max is None else float(lag_max)
        if lag_max < lag_min:
            raise ValueError("lag_max must be greater than or equal to lag_min")
        if n_eval is not None and n_eval < 1:
            raise ValueError("n_eval must be at least 1 when provided")
        if eval_delta <= 0:
            raise ValueError("eval_delta must be positive")

        if n_eval is None:
            pred_lags = np.arange(lag_min, lag_max + 1e-9, eval_delta, dtype=float)
            pred_lags = pred_lags[pred_lags <= lag_max + 1e-9]
            if pred_lags.size == 0:
                pred_lags = np.asarray([lag_min], dtype=float)
            eval_delta_out: float | None = float(eval_delta)
        else:
            pred_lags = np.linspace(lag_min, lag_max, int(n_eval), dtype=float)
            eval_delta_out = None
        return pred_lags, eval_delta_out

    def _build_named_regressor_rows(
        self,
        regressor: str,
        pred_lags: np.ndarray,
    ) -> np.ndarray:
        if regressor not in self.regressor_slices_:
            raise ValueError(
                f"No regressor slice found for {regressor!r}. Provide regressor_slices in fit()."
            )
        if regressor not in self.lag_design_infos_:
            raise ValueError(
                f"No lag design info found for {regressor!r}. Provide lag_design_infos in fit()."
            )

        basis_rows = np.asarray(
            dmatrix(
                self.lag_design_infos_[regressor],
                {"x": pred_lags.reshape(-1, 1)},
            )
        )
        effect_rows = np.zeros((pred_lags.size, self.n_regressors_), dtype=float)
        regressor_slice = self.regressor_slices_[regressor]
        if basis_rows.shape[1] != effect_rows[:, regressor_slice].shape[1]:
            raise ValueError(
                f"Lag basis column count for {regressor!r} does not match its registered slice."
            )
        effect_rows[:, regressor_slice] = basis_rows
        return effect_rows

    def _finalize_from_partial_state(self) -> "DistributedLagEchoModel":
        """Solve the shared penalized system from accumulated sufficient statistics."""
        n_regressors = self._partial_n_regressors
        n_voxels = self._partial_n_voxels
        total_nuisance = self._partial_CtC.shape[0]

        # Lift the time-domain cross-products into the full echo-wise system. The
        # Kronecker product with I_E means every time-domain coefficient gets a
        # separate parameter at each echo.
        effect_normal = kron(
            csc_matrix(self._partial_XtX), eye(self.E, format="csc"), format="csc"
        )
        cross_normal = kron(
            csc_matrix(self._partial_XtC), eye(self.E, format="csc"), format="csc"
        )
        nuisance_normal = kron(
            csc_matrix(self._partial_CtC), eye(self.E, format="csc"), format="csc"
        )

        # Each regressor-of-interest has an echo coefficient vector. Penalize its
        # overall magnitude with ridge and its roughness across neighboring echoes
        # with the second-difference penalty.
        effect_echo_penalty = ridge_penalty(
            self.E, self.ridge_alpha
        ) + second_difference_penalty(self.E, self.echo_smooth_alpha)
        effect_penalty = kron(
            eye(n_regressors, format="csc"), effect_echo_penalty, format="csc"
        )
        effect_param_count = n_regressors * self.E
        nuisance_param_count = total_nuisance * self.E
        if self.nuisance_ridge_alpha > 0:
            nuisance_penalty = kron(
                eye(total_nuisance, format="csc"),
                ridge_penalty(self.E, self.nuisance_ridge_alpha),
                format="csc",
            )
        else:
            nuisance_penalty = zero_penalty(nuisance_param_count)

        full_normal = vstack(
            [
                hstack([effect_normal, cross_normal], format="csc"),
                hstack([cross_normal.T, nuisance_normal], format="csc"),
            ],
            format="csc",
        )
        full_penalty = vstack(
            [
                hstack(
                    [
                        effect_penalty,
                        csc_matrix((effect_param_count, nuisance_param_count)),
                    ],
                    format="csc",
                ),
                hstack(
                    [
                        csc_matrix((nuisance_param_count, effect_param_count)),
                        nuisance_penalty,
                    ],
                    format="csc",
                ),
            ],
            format="csc",
        )

        rhs = np.vstack(
            [
                self._partial_XtY.reshape(effect_param_count, n_voxels),
                self._partial_CtY.reshape(nuisance_param_count, n_voxels),
            ]
        )
        # Solve one shared penalized linear system for all voxels at once. Columns of
        # rhs correspond to voxels; rows correspond to stacked effect and nuisance
        # parameters across echoes.
        self._solver = QuadraticPenaltySolver(
            as_csc(full_normal),
            as_csc(full_penalty),
        )
        del full_normal, full_penalty
        beta = self._solver.solve(rhs)
        del rhs

        effect_beta = beta[:effect_param_count].T.reshape(
            n_voxels, n_regressors, self.E
        )
        nuisance_beta = beta[effect_param_count:].T.reshape(
            n_voxels, total_nuisance, self.E
        )
        del beta

        self.effect_coefs_ = effect_beta
        self.nuisance_coefs_ = nuisance_beta
        self.n_regressors_ = n_regressors
        self.n_nuisance_ = total_nuisance
        self.n_voxels_ = n_voxels
        self.intercept_column_indices_ = list(self._partial_intercept_column_indices)
        self.run_regressor_designs_ = list(self._partial_run_regressor_designs)
        self.run_nuisance_designs_ = list(self._partial_run_nuisance_designs)
        self.run_nuisance_column_indices_ = [
            indices.copy() for indices in self._partial_run_nuisance_column_indices
        ]
        self.regressor_slices_ = dict(self._partial_regressor_slices)
        self.lag_design_infos_ = dict(self._partial_lag_design_infos)

        self._clear_partial_fit_state()
        return self

    def fit(
        self,
        regressors: list[np.ndarray],
        data: list[np.ndarray],
        confounds: list[np.ndarray] | None = None,
        regressor_slices: Mapping[str, slice] | None = None,
        lag_design_infos: Mapping[str, Any] | None = None,
    ) -> "DistributedLagEchoModel":
        """Fit the TE-dependent distributed lag model.

        Parameters
        ----------
        regressors : list[np.ndarray]
                One design matrix per run with shape ``time x regressors``. These are the
                regressors of interest, typically already expanded into a lag basis.
        data : list[np.ndarray]
                One multi-echo data tensor per run with shape ``voxel x echo x time``.
        confounds : list[np.ndarray] | None
                Optional nuisance regressors per run with shape ``time x nuisance``.
                Run-specific intercepts are always added automatically.

                .. warning::
                    All runs must supply the **same number of confound columns**, and
                    the model estimates a **single shared coefficient** per confound
                    column across all runs. This is appropriate for basis functions
                    that have identical meaning in every run (e.g., a DCT drift basis
                    with the same frequency grid). It is **not appropriate** for
                    run-specific regressors such as head motion parameters or
                    aCompCor components, whose scale and BOLD coupling vary across
                    runs. Pass such regressors through a separate denoising step
                    before calling ``fit``.
        regressor_slices : Mapping[str, slice] | None
                Optional mapping from regressor names to column slices in the design
            matrix. This is used by ``predict_curve_across_lags``.
        lag_design_infos : Mapping[str, Any] | None
                Optional Patsy ``design_info`` objects keyed by regressor name. These are
                used to evaluate the lag basis for prediction at arbitrary lag values.
        """
        if len(regressors) == 0 or len(data) == 0:
            raise ValueError("regressors and data must be non-empty lists")
        if len(regressors) != len(data):
            raise ValueError("regressors and data must have the same number of runs")
        if confounds is not None and len(confounds) != len(data):
            raise ValueError("confounds and data must have the same number of runs")

        confounds_list = [None] * len(data) if confounds is None else list(confounds)
        self._clear_fit_results()
        self._clear_partial_fit_state()
        if confounds is None:
            warnings.warn(
                "No confounds were provided; low-frequency trends are not removed from the TE-dependent model.",
                stacklevel=2,
            )

        for run_index, (run_design, run_data, run_confounds) in enumerate(
            zip(regressors, data, confounds_list)
        ):
            self.fit_partial(
                run_design,
                run_data,
                confounds=run_confounds,
                regressor_slices=regressor_slices if run_index == 0 else None,
                lag_design_infos=lag_design_infos if run_index == 0 else None,
            )

        return self.finalize_fit()

    def fit_partial(
        self,
        regressors: np.ndarray,
        data: np.ndarray,
        confounds: np.ndarray | None = None,
        regressor_slices: Mapping[str, slice] | None = None,
        lag_design_infos: Mapping[str, Any] | None = None,
    ) -> "DistributedLagEchoModel":
        """Accumulate one run's sufficient statistics for a later final solve.

        Parameters
        ----------
        regressors : np.ndarray
            Design matrix for this run with shape ``time x regressors``.
        data : np.ndarray
            Multi-echo data tensor for this run with shape ``voxel x echo x time``.
        confounds : np.ndarray | None
            Optional nuisance matrix for this run with shape ``time x nuisance``.
            A run-specific intercept column is always appended automatically.

            .. warning::
                The column count must match every previously supplied confound
                matrix. Each column is treated as a **shared regressor**: its
                cross-products are pooled across runs and a single coefficient
                is estimated for all runs jointly. This is valid only for
                regressors that have the same interpretation and comparable
                variance in every run (e.g., a fixed-frequency DCT drift basis).
                Run-varying nuisance regressors (motion parameters, aCompCor,
                FD/DVARS) should be regressed out of the data before calling
                this method.
        regressor_slices : Mapping[str, slice] | None
            Column slices for named regressors. Only needed on the first call;
            ignored (and consistency-checked) on subsequent calls.
        lag_design_infos : Mapping[str, Any] | None
            Patsy design_info objects for lag-basis prediction. Only needed on
            the first call.
        """
        run_design_array = np.asarray(regressors, dtype=float)
        run_data_array = np.asarray(data, dtype=float)
        if run_design_array.ndim != 2:
            raise ValueError("Each regressor design must be 2D (time x regressors)")
        if run_data_array.ndim != 3:
            raise ValueError("Each data tensor must be 3D (voxel x echo x time)")
        if run_data_array.shape[1] != self.E:
            raise ValueError(
                f"Expected {self.E} echoes, got {run_data_array.shape[1]} in one run"
            )
        if run_design_array.shape[0] != run_data_array.shape[2]:
            raise ValueError(
                "Each run design must have the same number of timepoints as the corresponding data tensor"
            )
        if np.any(~np.isfinite(run_design_array)):
            raise ValueError("Regressor designs must be finite")
        if np.any(~np.isfinite(run_data_array)):
            raise ValueError("Data tensors must be finite")

        run_confounds_array = self._normalize_run_confounds(
            confounds,
            n_timepoints=run_design_array.shape[0],
        )
        shared_confound_count = (
            0 if run_confounds_array is None else run_confounds_array.shape[1]
        )

        if not hasattr(self, "_partial_run_count"):
            if shared_confound_count == 0:
                warnings.warn(
                    "No confounds were provided; low-frequency trends are not removed from the TE-dependent model.",
                    stacklevel=2,
                )
            self._initialize_partial_fit_state(
                n_voxels=run_data_array.shape[0],
                n_regressors=run_design_array.shape[1],
                shared_confound_count=shared_confound_count,
                regressor_slices=regressor_slices,
                lag_design_infos=lag_design_infos,
            )
        else:
            if run_data_array.shape[0] != self._partial_n_voxels:
                raise ValueError("All data tensors must have the same voxel dimension")
            if run_design_array.shape[1] != self._partial_n_regressors:
                raise ValueError(
                    "All regressor designs must have the same number of columns"
                )
            if shared_confound_count != self._partial_shared_confound_count:
                raise ValueError(
                    "All confound matrices must have the same number of columns across runs"
                )
            self._store_prediction_metadata(regressor_slices, lag_design_infos)

        intercept_index = self._append_intercept_column()
        if run_confounds_array is None:
            run_nuisance_design = np.ones((run_design_array.shape[0], 1), dtype=float)
            nuisance_column_indices = np.asarray([intercept_index], dtype=int)
        else:
            run_nuisance_design = np.column_stack(
                [run_confounds_array, np.ones(run_design_array.shape[0], dtype=float)]
            )
            nuisance_column_indices = np.concatenate(
                [
                    np.arange(self._partial_shared_confound_count, dtype=int),
                    np.asarray([intercept_index], dtype=int),
                ]
            )

        # Accumulate the normal-equation pieces directly from this run. This is
        # algebraically equivalent to building one giant stacked design/data matrix,
        # but avoids duplicating the full multi-echo dataset in memory.
        self._partial_XtX += run_design_array.T @ run_design_array
        self._partial_XtC[:, nuisance_column_indices] += (
            run_design_array.T @ run_nuisance_design
        )
        self._partial_CtC[np.ix_(nuisance_column_indices, nuisance_column_indices)] += (
            run_nuisance_design.T @ run_nuisance_design
        )
        # run_data is stored as voxel x echo x time. Contract only over time so each
        # regressor or nuisance column gets one coefficient per echo, voxel.
        self._partial_XtY += np.einsum("tk,vet->kev", run_design_array, run_data_array)
        self._partial_CtY[nuisance_column_indices] += np.einsum(
            "tj,vet->jev", run_nuisance_design, run_data_array
        )

        self._partial_run_regressor_designs.append(run_design_array)
        self._partial_run_nuisance_designs.append(run_nuisance_design)
        self._partial_run_nuisance_column_indices.append(nuisance_column_indices)
        return self

    def finalize_fit(self) -> "DistributedLagEchoModel":
        """Solve the penalized TE model after one or more partial-fit updates."""
        if not hasattr(self, "_partial_run_count") or self._partial_run_count == 0:
            raise ValueError("No partial fit state available. Call fit_partial first.")
        return self._finalize_from_partial_state()

    def predict_curve(
        self,
        regressor_values: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> EchoCurvePrediction:
        """Predict the echo dependence for a specified regressor pattern.

        Parameters
        ----------
        regressor_values : np.ndarray
                Vector of regressor values with length equal to the number of columns in
                the effect design matrix. For a spline-lag basis this is usually a basis
                row evaluated at a chosen lag.
        pred_val : float
                Scalar multiplier applied to the regressor vector.
        run : int
                Run index whose intercept term should be used for prediction.
        include_intercept : bool
                If True, include the selected run intercept in the nuisance prediction.
        nuisance_values : np.ndarray | None
                Optional nuisance vector with length equal to the total number of nuisance
                columns. When omitted, nuisance terms are set to zero except for the run
                intercept if ``include_intercept`` is True.
        """
        if not hasattr(self, "effect_coefs_"):
            raise ValueError("Model must be fit before prediction")
        if run < 0 or run >= len(self.intercept_column_indices_):
            raise ValueError(
                f"run must be between 0 and {len(self.intercept_column_indices_) - 1}"
            )

        design_row = np.asarray(regressor_values, dtype=float).reshape(-1)
        if design_row.shape[0] != self.n_regressors_:
            raise ValueError(
                f"regressor_values must have length {self.n_regressors_}, got {design_row.shape[0]}"
            )

        if nuisance_values is None:
            nuisance_row = np.zeros(self.n_nuisance_, dtype=float)
        else:
            nuisance_row = np.asarray(nuisance_values, dtype=float).reshape(-1)
            if nuisance_row.shape[0] != self.n_nuisance_:
                raise ValueError(
                    f"nuisance_values must have length {self.n_nuisance_}, got {nuisance_row.shape[0]}"
                )
        if include_intercept:
            nuisance_row[self.intercept_column_indices_[run]] = 1.0

        # Contract the requested design row with the fitted echo-wise coefficient
        # curves. The result is one predicted TE curve per voxel.
        pred_effect = pred_val * np.einsum("k,vke->ve", design_row, self.effect_coefs_)
        pred_nuisance = np.einsum("j,vje->ve", nuisance_row, self.nuisance_coefs_)
        pred_curve = pred_effect + pred_nuisance

        return EchoCurvePrediction(
            echo_times_ms=self.echo_times_ms.copy(),
            pred_curve=pred_curve.T,
            pred_effect=pred_effect.T,
            pred_nuisance=pred_nuisance.T,
            params=EchoCurvePredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
            ),
        )

    def predict_curve_across_lags(
        self,
        regressor: str,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        lags: np.ndarray | list[float] | None = None,
        lag_min: float | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = 1.0,
    ) -> EchoCurveAcrossLagsPrediction:
        """Predict TE curves for one named regressor across a sampled lag grid.

        Returns
        -------
        EchoCurveAcrossLagsPrediction
            Prediction container with arrays shaped ``lag x echo x voxel``.
        """
        if not hasattr(self, "effect_coefs_"):
            raise ValueError("Model must be fit before prediction")

        pred_lags, eval_delta_out = self._build_prediction_lag_grid(
            regressor=regressor,
            lags=lags,
            lag_min=lag_min,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_named_regressor_rows(regressor, pred_lags)

        nuisance_prediction = self.predict_curve(
            regressor_values=np.zeros(self.n_regressors_, dtype=float),
            pred_val=0.0,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
        )
        nuisance_curve = nuisance_prediction.pred_nuisance.T

        pred_effect = pred_val * np.einsum(
            "lk,vke->lve", effect_rows, self.effect_coefs_
        )
        pred_nuisance = np.broadcast_to(nuisance_curve[None, :, :], pred_effect.shape)
        pred_curve = pred_effect + pred_nuisance

        return EchoCurveAcrossLagsPrediction(
            echo_times_ms=self.echo_times_ms.copy(),
            pred_lags=pred_lags,
            pred_curve=np.transpose(pred_curve, (0, 2, 1)),
            pred_effect=np.transpose(pred_effect, (0, 2, 1)),
            pred_nuisance=np.transpose(pred_nuisance, (0, 2, 1)),
            params=EchoCurveAcrossLagsPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=regressor,
                lag_min=float(pred_lags[0]),
                lag_max=float(pred_lags[-1]),
                n_eval=None if n_eval is None and lags is None else int(pred_lags.size),
                eval_delta=eval_delta_out,
            ),
        )

    def predict_curve_across_lags_for_echo(
        self,
        regressor: str,
        echo_index: int,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        lags: np.ndarray | list[float] | None = None,
        lag_min: float | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = 1.0,
    ) -> EchoCurveAcrossLagsForEchoPrediction:
        """Predict one echo's lag profile for a named regressor.

        Returns
        -------
        EchoCurveAcrossLagsForEchoPrediction
            Prediction container with arrays shaped ``lag x voxel``.
        """
        if echo_index < 0 or echo_index >= self.E:
            raise ValueError(f"echo_index must be between 0 and {self.E - 1}")

        across_lags = self.predict_curve_across_lags(
            regressor=regressor,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=lags,
            lag_min=lag_min,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )

        return EchoCurveAcrossLagsForEchoPrediction(
            pred_lags=across_lags.pred_lags.copy(),
            pred_curve=across_lags.pred_curve[:, echo_index, :],
            pred_effect=across_lags.pred_effect[:, echo_index, :],
            pred_nuisance=across_lags.pred_nuisance[:, echo_index, :],
            params=EchoCurveAcrossLagsForEchoPredictionParams(
                echo_index=echo_index,
                echo_time_ms=float(self.echo_times_ms[echo_index]),
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=regressor,
                lag_min=across_lags.params.lag_min,
                lag_max=across_lags.params.lag_max,
                n_eval=across_lags.params.n_eval,
                eval_delta=across_lags.params.eval_delta,
            ),
        )

    def predict_curve_at_lag(
        self,
        regressor: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> EchoCurvePrediction:
        """Predict the echo dependence for one named regressor at a chosen lag.

        This is a compatibility wrapper around ``predict_curve_across_lags`` for the
        common single-lag case.
        """
        across_lags = self.predict_curve_across_lags(
            regressor=regressor,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return EchoCurvePrediction(
            echo_times_ms=across_lags.echo_times_ms.copy(),
            pred_curve=across_lags.pred_curve[0],
            pred_effect=across_lags.pred_effect[0],
            pred_nuisance=across_lags.pred_nuisance[0],
            params=EchoCurvePredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=regressor,
                lag=float(lag),
            ),
        )

    def predict_timecourse_for_echo(
        self,
        echo_index: int,
        run: int = 0,
        include_nuisance: bool = True,
    ) -> EchoTimecoursePrediction:
        """Predict the fitted timecourse across time for one selected echo.

        Parameters
        ----------
        echo_index : int
            Zero-based echo index to extract from the fitted model.
        run : int
            Run index for which to compute the predicted timecourse.
        include_nuisance : bool
            If True, include the fitted nuisance contribution for the selected run.
            If False, return only the contribution of the regressors of interest.
        """
        if not hasattr(self, "effect_coefs_"):
            raise ValueError("Model must be fit before prediction")
        if echo_index < 0 or echo_index >= self.E:
            raise ValueError(f"echo_index must be between 0 and {self.E - 1}")
        if run < 0 or run >= len(self.run_regressor_designs_):
            raise ValueError(
                f"run must be between 0 and {len(self.run_regressor_designs_) - 1}"
            )

        run_effect_design = self.run_regressor_designs_[run]
        run_nuisance_design = self.run_nuisance_designs_[run]
        run_nuisance_column_indices = self.run_nuisance_column_indices_[run]
        effect_coefs = self.effect_coefs_[:, :, echo_index]
        # Fix one echo, then evaluate the usual time-domain GLM prediction using only
        # that echo's coefficient slice.
        pred_effect = np.einsum("tk,vk->tv", run_effect_design, effect_coefs)

        nuisance_coefs = self.nuisance_coefs_[
            :, run_nuisance_column_indices, echo_index
        ]
        pred_nuisance = np.einsum("tj,vj->tv", run_nuisance_design, nuisance_coefs)
        pred_timecourse = (
            pred_effect + pred_nuisance if include_nuisance else pred_effect
        )

        return EchoTimecoursePrediction(
            pred_timecourse=pred_timecourse,
            pred_effect=pred_effect,
            pred_nuisance=pred_nuisance,
            params=EchoTimecoursePredictionParams(
                echo_index=echo_index,
                echo_time_ms=float(self.echo_times_ms[echo_index]),
                run=run,
                include_nuisance=include_nuisance,
            ),
        )
