"""Shared estimation core for monoexponential multi-echo distributed lag models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.sparse import csc_matrix, eye, hstack, kron, vstack

from task_arousal.quadratic_penalty import (
    QuadraticPenaltySolver,
    as_csc,
    ridge_penalty,
    zero_penalty,
)


@dataclass
class MonoexponentialCurvePredictionParams:
    pred_val: float
    run: int
    include_intercept: bool
    regressor: str | None = None
    lag: float | None = None


@dataclass
class MonoexponentialCurvePrediction:
    echo_times_ms: np.ndarray
    pred_curve: np.ndarray
    pred_log_curve: np.ndarray
    pred_log_effect: np.ndarray
    pred_log_nuisance: np.ndarray
    params: MonoexponentialCurvePredictionParams


@dataclass
class MonoexponentialCurveAcrossLagsPredictionParams:
    pred_val: float
    run: int
    include_intercept: bool
    regressor: str
    lag_min: float
    lag_max: float
    n_eval: int | None
    eval_delta: float | None


@dataclass
class MonoexponentialCurveAcrossLagsPrediction:
    echo_times_ms: np.ndarray
    pred_lags: np.ndarray
    pred_curve: np.ndarray
    pred_log_curve: np.ndarray
    pred_log_effect: np.ndarray
    pred_log_nuisance: np.ndarray
    params: MonoexponentialCurveAcrossLagsPredictionParams


@dataclass
class MonoexponentialCurveAcrossLagsForEchoPredictionParams:
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
class MonoexponentialCurveAcrossLagsForEchoPrediction:
    pred_lags: np.ndarray
    pred_curve: np.ndarray
    pred_log_curve: np.ndarray
    pred_log_effect: np.ndarray
    pred_log_nuisance: np.ndarray
    params: MonoexponentialCurveAcrossLagsForEchoPredictionParams


@dataclass
class MonoexponentialParamPredictionParams:
    pred_val: float
    run: int
    include_intercept: bool
    parameterization: str
    regressor: str | None = None
    lag: float | None = None


@dataclass
class MonoexponentialParamPrediction:
    param_names: tuple[str, ...]
    pred_params: np.ndarray
    pred_effect_params: np.ndarray
    pred_nuisance_params: np.ndarray
    params: MonoexponentialParamPredictionParams


@dataclass
class MonoexponentialParamAcrossLagsPredictionParams:
    pred_val: float
    run: int
    include_intercept: bool
    parameterization: str
    regressor: str
    lag_min: float
    lag_max: float
    n_eval: int | None
    eval_delta: float | None


@dataclass
class MonoexponentialParamAcrossLagsPrediction:
    pred_lags: np.ndarray
    param_names: tuple[str, ...]
    pred_params: np.ndarray
    pred_effect_params: np.ndarray
    pred_nuisance_params: np.ndarray
    params: MonoexponentialParamAcrossLagsPredictionParams


@dataclass
class MonoexponentialDerivedParamPrediction:
    param_names: tuple[str, ...]
    pred_params: np.ndarray
    params: MonoexponentialParamPredictionParams


@dataclass
class MonoexponentialDerivedParamAcrossLagsPrediction:
    pred_lags: np.ndarray
    param_names: tuple[str, ...]
    pred_params: np.ndarray
    params: MonoexponentialParamAcrossLagsPredictionParams


class _DistributedLagMonoexponentialEchoBase:
    """Shared run-wise estimator for monoexponential multi-echo lag models."""

    def __init__(
        self,
        echo_times_ms: np.ndarray | list[float],
        ridge_alpha: float = 1.0,
        nuisance_ridge_alpha: float = 0.0,
        te_rescale_factor: float = 10.0,
        min_signal: float = 1e-6,
        max_t2star_ms: float = 500.0,
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
        if ridge_alpha < 0 or nuisance_ridge_alpha < 0:
            raise ValueError("Penalty weights must be non-negative")
        if te_rescale_factor <= 0:
            raise ValueError("te_rescale_factor must be strictly positive")
        if min_signal <= 0:
            raise ValueError("min_signal must be strictly positive")
        if max_t2star_ms <= 0:
            raise ValueError("max_t2star_ms must be strictly positive")

        self.E = len(self.echo_times_ms)
        self.ridge_alpha = float(ridge_alpha)
        self.nuisance_ridge_alpha = float(nuisance_ridge_alpha)
        self.te_rescale_factor = float(te_rescale_factor)
        self.min_signal = float(min_signal)
        self.max_t2star_ms = float(max_t2star_ms)

        self.echo_times_scaled_ = self.echo_times_ms / self.te_rescale_factor
        self._te_design = np.column_stack(
            [np.ones(self.E, dtype=float), -self.echo_times_scaled_]
        )
        self._te_gram = self._te_design.T @ self._te_design

    def _clear_fit_results(self) -> None:
        for attr in (
            "effect_loglin_coefs_",
            "nuisance_loglin_coefs_",
            "n_regressors_",
            "n_nuisance_",
            "n_voxels_",
            "intercept_column_indices_",
            "run_regressor_designs_",
            "run_nuisance_designs_",
            "run_nuisance_column_indices_",
            "_solver",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _clear_partial_fit_state(self) -> None:
        for attr in (
            "_partial_n_voxels",
            "_partial_n_regressors",
            "_partial_shared_confound_count",
            "_partial_run_count",
            "_partial_XtX",
            "_partial_XtC",
            "_partial_CtC",
            "_partial_XtY_loglin",
            "_partial_CtY_loglin",
            "_partial_intercept_column_indices",
            "_partial_run_regressor_designs",
            "_partial_run_nuisance_designs",
            "_partial_run_nuisance_column_indices",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _normalize_run_confounds(
        self,
        run_confounds: np.ndarray | None,
        n_timepoints: int,
    ) -> np.ndarray | None:
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
    ) -> None:
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
        self._partial_XtY_loglin = np.zeros((n_regressors, 2, n_voxels), dtype=float)
        self._partial_CtY_loglin = np.zeros(
            (shared_confound_count, 2, n_voxels), dtype=float
        )
        self._partial_intercept_column_indices: list[int] = []
        self._partial_run_regressor_designs: list[np.ndarray] = []
        self._partial_run_nuisance_designs: list[np.ndarray] = []
        self._partial_run_nuisance_column_indices: list[np.ndarray] = []

    def _validate_run_arrays(
        self,
        run_design_array: np.ndarray,
        run_data_array: np.ndarray,
    ) -> None:
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

    def _validate_existing_partial_dimensions(
        self,
        run_design_array: np.ndarray,
        run_data_array: np.ndarray,
        shared_confound_count: int,
    ) -> None:
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

    def _append_intercept_column(self) -> int:
        self._partial_XtC = np.pad(self._partial_XtC, ((0, 0), (0, 1)))
        self._partial_CtC = np.pad(self._partial_CtC, ((0, 1), (0, 1)))
        self._partial_CtY_loglin = np.pad(
            self._partial_CtY_loglin, ((0, 1), (0, 0), (0, 0))
        )
        intercept_index = self._partial_shared_confound_count + self._partial_run_count
        self._partial_intercept_column_indices.append(intercept_index)
        self._partial_run_count += 1
        return intercept_index

    def _prepare_log_data(self, run_data_array: np.ndarray) -> np.ndarray:
        return np.log(np.clip(run_data_array, a_min=self.min_signal, a_max=None))

    def _accumulate_run_statistics(
        self,
        run_design_array: np.ndarray,
        run_data_array: np.ndarray,
        run_nuisance_design: np.ndarray,
        nuisance_column_indices: np.ndarray,
    ) -> None:
        self._partial_XtX += run_design_array.T @ run_design_array
        self._partial_XtC[:, nuisance_column_indices] += (
            run_design_array.T @ run_nuisance_design
        )
        self._partial_CtC[np.ix_(nuisance_column_indices, nuisance_column_indices)] += (
            run_nuisance_design.T @ run_nuisance_design
        )

        log_data = self._prepare_log_data(run_data_array)
        log_sum = np.sum(log_data, axis=1)
        te_weighted_log_sum = np.einsum("e,vet->vt", self.echo_times_scaled_, log_data)

        self._partial_XtY_loglin[:, 0] += run_design_array.T @ log_sum.T
        self._partial_XtY_loglin[:, 1] += run_design_array.T @ (-te_weighted_log_sum).T
        self._partial_CtY_loglin[nuisance_column_indices, 0] += (
            run_nuisance_design.T @ log_sum.T
        )
        self._partial_CtY_loglin[nuisance_column_indices, 1] += (
            run_nuisance_design.T @ (-te_weighted_log_sum).T
        )

        self._partial_run_regressor_designs.append(run_design_array)
        self._partial_run_nuisance_designs.append(run_nuisance_design)
        self._partial_run_nuisance_column_indices.append(nuisance_column_indices)

    def _finalize_prediction_metadata(self) -> None:
        """Hook for subclasses to copy prediction metadata from partial to fitted state."""

    def _finalize_from_partial_state(self):
        n_regressors = self._partial_n_regressors
        n_voxels = self._partial_n_voxels
        total_nuisance = self._partial_CtC.shape[0]
        te_gram = csc_matrix(self._te_gram)

        effect_normal = kron(csc_matrix(self._partial_XtX), te_gram, format="csc")
        cross_normal = kron(csc_matrix(self._partial_XtC), te_gram, format="csc")
        nuisance_normal = kron(csc_matrix(self._partial_CtC), te_gram, format="csc")

        effect_penalty = kron(
            eye(n_regressors, format="csc"),
            ridge_penalty(2, self.ridge_alpha),
            format="csc",
        )
        effect_param_count = n_regressors * 2
        nuisance_param_count = total_nuisance * 2
        if self.nuisance_ridge_alpha > 0:
            nuisance_penalty = kron(
                eye(total_nuisance, format="csc"),
                ridge_penalty(2, self.nuisance_ridge_alpha),
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
                self._partial_XtY_loglin.reshape(effect_param_count, n_voxels),
                self._partial_CtY_loglin.reshape(nuisance_param_count, n_voxels),
            ]
        )
        self._solver = QuadraticPenaltySolver(as_csc(full_normal), as_csc(full_penalty))
        beta = self._solver.solve(rhs)

        effect_beta = beta[:effect_param_count].T.reshape(n_voxels, n_regressors, 2)
        nuisance_beta = beta[effect_param_count:].T.reshape(n_voxels, total_nuisance, 2)

        self.effect_loglin_coefs_ = effect_beta
        self.nuisance_loglin_coefs_ = nuisance_beta
        self.n_regressors_ = n_regressors
        self.n_nuisance_ = total_nuisance
        self.n_voxels_ = n_voxels
        self.intercept_column_indices_ = list(self._partial_intercept_column_indices)
        self.run_regressor_designs_ = list(self._partial_run_regressor_designs)
        self.run_nuisance_designs_ = list(self._partial_run_nuisance_designs)
        self.run_nuisance_column_indices_ = [
            indices.copy() for indices in self._partial_run_nuisance_column_indices
        ]
        self._finalize_prediction_metadata()
        self._clear_partial_fit_state()
        return self

    def finalize_fit(self):
        if not hasattr(self, "_partial_run_count") or self._partial_run_count == 0:
            raise ValueError("No partial fit state available. Call fit_partial first.")
        return self._finalize_from_partial_state()

    def _build_nuisance_row(
        self,
        run: int,
        include_intercept: bool,
        nuisance_values: np.ndarray | None,
    ) -> np.ndarray:
        if run < 0 or run >= len(self.intercept_column_indices_):
            raise ValueError(
                f"run must be between 0 and {len(self.intercept_column_indices_) - 1}"
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
        return nuisance_row

    def _predict_loglinear_components(
        self,
        design_row: np.ndarray,
        pred_val: float,
        run: int,
        include_intercept: bool,
        nuisance_values: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "effect_loglin_coefs_"):
            raise ValueError("Model must be fit before prediction")

        design_row_array = np.asarray(design_row, dtype=float).reshape(-1)
        if design_row_array.shape[0] != self.n_regressors_:
            raise ValueError(
                f"regressor_values must have length {self.n_regressors_}, got {design_row_array.shape[0]}"
            )

        nuisance_row = self._build_nuisance_row(run, include_intercept, nuisance_values)
        pred_effect_params = pred_val * np.einsum(
            "k,vkp->vp", design_row_array, self.effect_loglin_coefs_
        )
        pred_nuisance_params = np.einsum(
            "j,vjp->vp", nuisance_row, self.nuisance_loglin_coefs_
        )
        pred_params = pred_effect_params + pred_nuisance_params
        return pred_params, pred_effect_params, pred_nuisance_params

    def _log_params_to_curve(
        self,
        log_params: np.ndarray,
    ) -> np.ndarray:
        intercept = log_params[..., 0]
        decay = log_params[..., 1]
        return intercept[..., None, :] - (
            self.echo_times_scaled_[None, :, None] * decay[..., None, :]
        )

    def _curve_components_from_loglinear(
        self,
        pred_effect_params: np.ndarray,
        pred_nuisance_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        effect_log = self._log_params_to_curve(pred_effect_params[None, ...])[0]
        nuisance_log = self._log_params_to_curve(pred_nuisance_params[None, ...])[0]
        pred_log_curve = effect_log + nuisance_log
        pred_curve = np.exp(pred_log_curve)
        return pred_curve, pred_log_curve, effect_log, nuisance_log

    def _curve_components_from_rows(
        self,
        pred_effect_params: np.ndarray,
        pred_nuisance_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        intercept_effect = pred_effect_params[..., 0]
        decay_effect = pred_effect_params[..., 1]
        intercept_nuisance = pred_nuisance_params[..., 0]
        decay_nuisance = pred_nuisance_params[..., 1]
        effect_log = intercept_effect[:, None, :] - (
            self.echo_times_scaled_[None, :, None] * decay_effect[:, None, :]
        )
        nuisance_log = intercept_nuisance[:, None, :] - (
            self.echo_times_scaled_[None, :, None] * decay_nuisance[:, None, :]
        )
        pred_log_curve = effect_log + nuisance_log
        pred_curve = np.exp(pred_log_curve)
        return pred_curve, pred_log_curve, effect_log, nuisance_log

    def _derive_t2star_params(self, pred_params: np.ndarray) -> np.ndarray:
        derived = np.empty_like(pred_params)
        derived[..., 0] = pred_params[..., 0]
        derived[..., 1] = np.nan
        np.divide(
            self.te_rescale_factor,
            pred_params[..., 1],
            out=derived[..., 1],
            where=pred_params[..., 1] > 0,
        )
        np.clip(
            derived[..., 1], a_min=None, a_max=self.max_t2star_ms, out=derived[..., 1]
        )
        return derived

    def predict_loglinear_params(
        self,
        regressor_values: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialParamPrediction:
        pred_params, pred_effect_params, pred_nuisance_params = (
            self._predict_loglinear_components(
                design_row=regressor_values,
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                nuisance_values=nuisance_values,
            )
        )
        return MonoexponentialParamPrediction(
            param_names=("intercept", "decay_rate"),
            pred_params=pred_params.T,
            pred_effect_params=pred_effect_params.T,
            pred_nuisance_params=pred_nuisance_params.T,
            params=MonoexponentialParamPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="loglinear",
            ),
        )

    def predict_t2star_params(
        self,
        regressor_values: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialDerivedParamPrediction:
        pred_params, _, _ = self._predict_loglinear_components(
            design_row=regressor_values,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
        )
        derived = self._derive_t2star_params(pred_params)
        return MonoexponentialDerivedParamPrediction(
            param_names=("intercept", "t2star_ms"),
            pred_params=derived.T,
            params=MonoexponentialParamPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="t2star",
            ),
        )

    def predict_curve(
        self,
        regressor_values: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialCurvePrediction:
        pred_params, pred_effect_params, pred_nuisance_params = (
            self._predict_loglinear_components(
                design_row=regressor_values,
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                nuisance_values=nuisance_values,
            )
        )
        pred_curve, pred_log_curve, pred_log_effect, pred_log_nuisance = (
            self._curve_components_from_loglinear(
                pred_effect_params=pred_effect_params,
                pred_nuisance_params=pred_nuisance_params,
            )
        )
        return MonoexponentialCurvePrediction(
            echo_times_ms=self.echo_times_ms.copy(),
            pred_curve=pred_curve,
            pred_log_curve=pred_log_curve,
            pred_log_effect=pred_log_effect,
            pred_log_nuisance=pred_log_nuisance,
            params=MonoexponentialCurvePredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
            ),
        )

    def _predict_loglinear_from_rows(
        self,
        regressor_name: str,
        pred_lags: np.ndarray,
        effect_rows: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        eval_delta: float | None = None,
        n_eval: int | None = None,
    ) -> MonoexponentialParamAcrossLagsPrediction:
        nuisance_row = self._build_nuisance_row(run, include_intercept, nuisance_values)
        pred_effect = pred_val * np.einsum(
            "lk,vkp->lvp", effect_rows, self.effect_loglin_coefs_
        )
        pred_nuisance = np.einsum(
            "j,vjp->vp", nuisance_row, self.nuisance_loglin_coefs_
        )
        pred_nuisance = np.broadcast_to(pred_nuisance[None, :, :], pred_effect.shape)
        pred_params = pred_effect + pred_nuisance
        return MonoexponentialParamAcrossLagsPrediction(
            pred_lags=pred_lags,
            param_names=("intercept", "decay_rate"),
            pred_params=np.transpose(pred_params, (0, 2, 1)),
            pred_effect_params=np.transpose(pred_effect, (0, 2, 1)),
            pred_nuisance_params=np.transpose(pred_nuisance, (0, 2, 1)),
            params=MonoexponentialParamAcrossLagsPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="loglinear",
                regressor=regressor_name,
                lag_min=float(pred_lags[0]),
                lag_max=float(pred_lags[-1]),
                n_eval=n_eval,
                eval_delta=eval_delta,
            ),
        )

    def _predict_t2star_from_rows(
        self,
        regressor_name: str,
        pred_lags: np.ndarray,
        effect_rows: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        eval_delta: float | None = None,
        n_eval: int | None = None,
    ) -> MonoexponentialDerivedParamAcrossLagsPrediction:
        loglinear = self._predict_loglinear_from_rows(
            regressor_name=regressor_name,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta,
            n_eval=n_eval,
        )
        pred_params = np.transpose(loglinear.pred_params, (0, 2, 1))
        derived = self._derive_t2star_params(pred_params)
        return MonoexponentialDerivedParamAcrossLagsPrediction(
            pred_lags=pred_lags,
            param_names=("intercept", "t2star_ms"),
            pred_params=np.transpose(derived, (0, 2, 1)),
            params=MonoexponentialParamAcrossLagsPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="t2star",
                regressor=regressor_name,
                lag_min=float(pred_lags[0]),
                lag_max=float(pred_lags[-1]),
                n_eval=n_eval,
                eval_delta=eval_delta,
            ),
        )

    def _predict_curve_from_rows(
        self,
        regressor_name: str,
        pred_lags: np.ndarray,
        effect_rows: np.ndarray,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        eval_delta: float | None = None,
        n_eval: int | None = None,
    ) -> MonoexponentialCurveAcrossLagsPrediction:
        nuisance_row = self._build_nuisance_row(run, include_intercept, nuisance_values)
        pred_effect = pred_val * np.einsum(
            "lk,vkp->lvp", effect_rows, self.effect_loglin_coefs_
        )
        pred_nuisance = np.einsum(
            "j,vjp->vp", nuisance_row, self.nuisance_loglin_coefs_
        )
        pred_nuisance = np.broadcast_to(pred_nuisance[None, :, :], pred_effect.shape)
        pred_curve, pred_log_curve, pred_log_effect, pred_log_nuisance = (
            self._curve_components_from_rows(
                pred_effect_params=pred_effect,
                pred_nuisance_params=pred_nuisance,
            )
        )
        return MonoexponentialCurveAcrossLagsPrediction(
            echo_times_ms=self.echo_times_ms.copy(),
            pred_lags=pred_lags,
            pred_curve=np.transpose(pred_curve, (0, 1, 2)),
            pred_log_curve=np.transpose(pred_log_curve, (0, 1, 2)),
            pred_log_effect=np.transpose(pred_log_effect, (0, 1, 2)),
            pred_log_nuisance=np.transpose(pred_log_nuisance, (0, 1, 2)),
            params=MonoexponentialCurveAcrossLagsPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=regressor_name,
                lag_min=float(pred_lags[0]),
                lag_max=float(pred_lags[-1]),
                n_eval=n_eval,
                eval_delta=eval_delta,
            ),
        )
