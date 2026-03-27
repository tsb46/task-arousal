"""Shared estimation core for multi-echo distributed lag models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


class _DistributedLagEchoBase:
    """Shared run-wise estimator and generic prediction helpers."""

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
            "_partial_XtY",
            "_partial_CtY",
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
        self._partial_XtY = np.zeros((n_regressors, self.E, n_voxels), dtype=float)
        self._partial_CtY = np.zeros(
            (shared_confound_count, self.E, n_voxels), dtype=float
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
        self._partial_CtY = np.pad(self._partial_CtY, ((0, 1), (0, 0), (0, 0)))
        intercept_index = self._partial_shared_confound_count + self._partial_run_count
        self._partial_intercept_column_indices.append(intercept_index)
        self._partial_run_count += 1
        return intercept_index

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
        self._partial_XtY += np.einsum("tk,vet->kev", run_design_array, run_data_array)
        self._partial_CtY[nuisance_column_indices] += np.einsum(
            "tj,vet->jev", run_nuisance_design, run_data_array
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

        effect_normal = kron(
            csc_matrix(self._partial_XtX), eye(self.E, format="csc"), format="csc"
        )
        cross_normal = kron(
            csc_matrix(self._partial_XtC), eye(self.E, format="csc"), format="csc"
        )
        nuisance_normal = kron(
            csc_matrix(self._partial_CtC), eye(self.E, format="csc"), format="csc"
        )

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
        self._finalize_prediction_metadata()
        self._clear_partial_fit_state()
        return self

    def finalize_fit(self):
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

    def _predict_from_rows(
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
    ) -> EchoCurveAcrossLagsPrediction:
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
                regressor=regressor_name,
                lag_min=float(pred_lags[0]),
                lag_max=float(pred_lags[-1]),
                n_eval=n_eval,
                eval_delta=eval_delta,
            ),
        )

    def predict_timecourse_for_echo(
        self,
        echo_index: int,
        run: int = 0,
        include_nuisance: bool = True,
    ) -> EchoTimecoursePrediction:
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
