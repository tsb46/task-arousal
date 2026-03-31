"""Monoexponential distributed lag modeling of multi-echo data."""

from __future__ import annotations

from typing import Any, Literal, Mapping
import warnings

import numpy as np
import pandas as pd

from patsy import dmatrix  # type: ignore
from scipy.interpolate import interp1d

from task_arousal.analysis._dlm_echo_mono_base import (
    MonoexponentialCurveAcrossLagsForEchoPrediction,
    MonoexponentialCurveAcrossLagsForEchoPredictionParams,
    MonoexponentialCurveAcrossLagsPrediction,
    MonoexponentialCurvePrediction,
    MonoexponentialCurvePredictionParams,
    MonoexponentialDerivedParamAcrossLagsPrediction,
    MonoexponentialDerivedParamPrediction,
    MonoexponentialParamAcrossLagsPrediction,
    MonoexponentialParamPrediction,
    MonoexponentialParamPredictionParams,
    _DistributedLagMonoexponentialEchoBase,
)
from task_arousal.analysis.basis import (
    SplineLagBasis,
    boxcar,
    create_spline_event_reg,
    normalize_run_regressors,
)
from task_arousal.analysis.dlm import PREDICT_T_DELTA, RESAMPLE_TR
from task_arousal.constants import EVENT_COLUMNS, SLICE_TIMING_REF


def _get_ordered_trial_types(event_dfs: list[pd.DataFrame]) -> list[str]:
    trial_types: list[str] = []
    seen: set[str] = set()
    for event_df in event_dfs:
        for trial in event_df["trial_type"].tolist():
            trial_name = str(trial)
            if trial_name in seen:
                continue
            seen.add(trial_name)
            trial_types.append(trial_name)
    return trial_types


def _build_trial_slices(
    trial_types: list[str],
    trial_bases: Mapping[str, SplineLagBasis],
) -> dict[str, slice]:
    trial_slices: dict[str, slice] = {}
    column_start = 0
    for trial in trial_types:
        n_trial_cols = int(np.asarray(trial_bases[trial].basis).shape[1])
        trial_slices[trial] = slice(column_start, column_start + n_trial_cols)
        column_start += n_trial_cols
    return trial_slices


class DistributedLagPhysioMonoexponentialEchoModel(
    _DistributedLagMonoexponentialEchoBase
):
    """Monoexponential multi-echo distributed lag model for a single physio regressor."""

    def __init__(
        self,
        echo_times_ms: np.ndarray | list[float],
        tr: float,
        nlags: int,
        neg_nlags: int = 0,
        knots_per_sec: float = 0.3,
        n_knots: int | None = None,
        knots: list[int] | None = None,
        basis_type: Literal["cr", "bs"] = "cr",
        regressor_name: str = "physio",
        ridge_alpha: float = 1.0,
        nuisance_ridge_alpha: float = 0.0,
        te_rescale_factor: float = 10.0,
        min_signal: float = 1e-6,
        max_t2star_ms: float = 500.0,
    ):
        super().__init__(
            echo_times_ms=echo_times_ms,
            ridge_alpha=ridge_alpha,
            nuisance_ridge_alpha=nuisance_ridge_alpha,
            te_rescale_factor=te_rescale_factor,
            min_signal=min_signal,
            max_t2star_ms=max_t2star_ms,
        )
        if tr <= 0:
            raise ValueError("tr must be positive")
        if nlags < 0:
            raise ValueError("nlags must be non-negative")
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        if regressor_name == "":
            raise ValueError("regressor_name must be non-empty")

        self.tr = float(tr)
        self.nlags = int(nlags)
        self.neg_nlags = int(neg_nlags)
        self.knots_per_sec = knots_per_sec
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type: Literal["cr", "bs"] = basis_type
        self.regressor_name = regressor_name

    def _clear_fit_results(self) -> None:
        super()._clear_fit_results()
        for attr in (
            "basis_",
            "regressor_name_",
            "regressor_slices_",
            "lag_design_infos_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _clear_partial_fit_state(self) -> None:
        super()._clear_partial_fit_state()
        for attr in (
            "_partial_basis",
            "_partial_regressor_slices",
            "_partial_lag_design_infos",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _create_basis(self) -> SplineLagBasis:
        basis = SplineLagBasis(
            knots_per_sec=self.knots_per_sec,
            tr=self.tr,
            n_knots=self.n_knots,
            knots=self.knots,
            basis_type=self.basis_type,
        )
        basis.create(self.nlags, self.neg_nlags)
        return basis

    def _normalize_physio_run(self, regressors: np.ndarray) -> np.ndarray:
        run_regressors = np.asarray(regressors, dtype=float)
        if run_regressors.ndim == 1:
            run_regressors = run_regressors.reshape(-1, 1)
        if run_regressors.ndim != 2 or run_regressors.shape[1] != 1:
            raise ValueError(
                "Each physio run must have shape (time, 1) or be one-dimensional"
            )
        if np.any(~np.isfinite(run_regressors)):
            raise ValueError("Physio regressors must be finite")
        return run_regressors

    def _initialize_prediction_metadata(self) -> None:
        basis = self._partial_basis
        self._partial_regressor_slices = {
            self.regressor_name: slice(0, int(np.asarray(basis.basis).shape[1]))
        }
        self._partial_lag_design_infos = {self.regressor_name: basis.basis.design_info}

    def _project_physio_run(self, regressors: np.ndarray) -> np.ndarray:
        run_regressors = self._normalize_physio_run(regressors)
        return np.asarray(self._partial_basis.project(run_regressors, fill_val=0.0))

    def _finalize_prediction_metadata(self) -> None:
        self.basis_ = getattr(self, "_partial_basis")
        self.regressor_name_ = self.regressor_name
        self.regressor_slices_ = dict(getattr(self, "_partial_regressor_slices", {}))
        self.lag_design_infos_ = dict(getattr(self, "_partial_lag_design_infos", {}))

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
    ) -> tuple[np.ndarray, float | None, int | None]:
        if lags is not None:
            pred_lags = np.asarray(lags, dtype=float).reshape(-1)
            if pred_lags.size == 0:
                raise ValueError("lags must contain at least one value")
            if np.any(~np.isfinite(pred_lags)):
                raise ValueError("lags must be finite")
            return pred_lags, None, int(pred_lags.size)

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
            return pred_lags, float(eval_delta), None

        pred_lags = np.linspace(lag_min, lag_max, int(n_eval), dtype=float)
        return pred_lags, None, int(n_eval)

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

    def fit(
        self,
        regressors: list[np.ndarray],
        data: list[np.ndarray],
        confounds: list[np.ndarray] | None = None,
    ) -> "DistributedLagPhysioMonoexponentialEchoModel":
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
                "No confounds were provided; low-frequency trends are not removed from the monoexponential TE-dependent physio model.",
                stacklevel=2,
            )
        self._partial_basis = self._create_basis()

        for run_regressors, run_data, run_confounds in zip(
            regressors, data, confounds_list
        ):
            self.fit_partial(run_regressors, run_data, confounds=run_confounds)

        return self.finalize_fit()

    def fit_partial(
        self,
        regressors: np.ndarray,
        data: np.ndarray,
        confounds: np.ndarray | None = None,
    ) -> "DistributedLagPhysioMonoexponentialEchoModel":
        basis = getattr(self, "_partial_basis", None)
        if basis is None:
            basis = self._create_basis()
            self._partial_basis = basis

        run_design_array = self._project_physio_run(regressors)
        run_data_array = np.asarray(data, dtype=float)
        self._validate_run_arrays(run_design_array, run_data_array)

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
                    "No confounds were provided; low-frequency trends are not removed from the monoexponential TE-dependent physio model.",
                    stacklevel=2,
                )
            self._initialize_partial_fit_state(
                n_voxels=run_data_array.shape[0],
                n_regressors=run_design_array.shape[1],
                shared_confound_count=shared_confound_count,
            )
            self._partial_basis = basis
            self._initialize_prediction_metadata()
        else:
            self._validate_existing_partial_dimensions(
                run_design_array,
                run_data_array,
                shared_confound_count,
            )

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

        self._accumulate_run_statistics(
            run_design_array,
            run_data_array,
            run_nuisance_design,
            nuisance_column_indices,
        )
        return self

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
    ) -> MonoexponentialCurveAcrossLagsPrediction:
        pred_lags, eval_delta_out, n_eval_out = self._build_prediction_lag_grid(
            regressor=regressor,
            lags=lags,
            lag_min=lag_min,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_named_regressor_rows(regressor, pred_lags)
        return self._predict_curve_from_rows(
            regressor_name=regressor,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta_out,
            n_eval=n_eval_out,
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
    ) -> MonoexponentialCurveAcrossLagsForEchoPrediction:
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
        return MonoexponentialCurveAcrossLagsForEchoPrediction(
            pred_lags=across_lags.pred_lags.copy(),
            pred_curve=across_lags.pred_curve[:, echo_index, :],
            pred_log_curve=across_lags.pred_log_curve[:, echo_index, :],
            pred_log_effect=across_lags.pred_log_effect[:, echo_index, :],
            pred_log_nuisance=across_lags.pred_log_nuisance[:, echo_index, :],
            params=MonoexponentialCurveAcrossLagsForEchoPredictionParams(
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
    ) -> MonoexponentialCurvePrediction:
        across_lags = self.predict_curve_across_lags(
            regressor=regressor,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return MonoexponentialCurvePrediction(
            echo_times_ms=across_lags.echo_times_ms.copy(),
            pred_curve=across_lags.pred_curve[0],
            pred_log_curve=across_lags.pred_log_curve[0],
            pred_log_effect=across_lags.pred_log_effect[0],
            pred_log_nuisance=across_lags.pred_log_nuisance[0],
            params=MonoexponentialCurvePredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=regressor,
                lag=float(lag),
            ),
        )

    def predict_loglinear_params_across_lags(
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
    ) -> MonoexponentialParamAcrossLagsPrediction:
        pred_lags, eval_delta_out, n_eval_out = self._build_prediction_lag_grid(
            regressor=regressor,
            lags=lags,
            lag_min=lag_min,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_named_regressor_rows(regressor, pred_lags)
        return self._predict_loglinear_from_rows(
            regressor_name=regressor,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta_out,
            n_eval=n_eval_out,
        )

    def predict_loglinear_params_at_lag(
        self,
        regressor: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialParamPrediction:
        across_lags = self.predict_loglinear_params_across_lags(
            regressor=regressor,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return MonoexponentialParamPrediction(
            param_names=across_lags.param_names,
            pred_params=across_lags.pred_params[0],
            pred_effect_params=across_lags.pred_effect_params[0],
            pred_nuisance_params=across_lags.pred_nuisance_params[0],
            params=MonoexponentialParamPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="loglinear",
                regressor=regressor,
                lag=float(lag),
            ),
        )

    def predict_t2star_across_lags(
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
    ) -> MonoexponentialDerivedParamAcrossLagsPrediction:
        pred_lags, eval_delta_out, n_eval_out = self._build_prediction_lag_grid(
            regressor=regressor,
            lags=lags,
            lag_min=lag_min,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_named_regressor_rows(regressor, pred_lags)
        return self._predict_t2star_from_rows(
            regressor_name=regressor,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta_out,
            n_eval=n_eval_out,
        )

    def predict_t2star_at_lag(
        self,
        regressor: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialDerivedParamPrediction:
        across_lags = self.predict_t2star_across_lags(
            regressor=regressor,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return MonoexponentialDerivedParamPrediction(
            param_names=across_lags.param_names,
            pred_params=across_lags.pred_params[0],
            params=MonoexponentialParamPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="t2star",
                regressor=regressor,
                lag=float(lag),
            ),
        )

    def evaluate(
        self,
        regressor: str,
        lag_max: float | None = None,
        lag_min: float | None = None,
        n_eval: int | None = None,
        pred_val: float = 1.0,
        eval_delta: float = PREDICT_T_DELTA,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialCurveAcrossLagsPrediction:
        return self.predict_curve_across_lags(
            regressor=regressor,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lag_min=lag_min,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )


class DistributedLagEventMonoexponentialEchoModel(
    _DistributedLagMonoexponentialEchoBase
):
    """Monoexponential multi-echo distributed lag model for task-event lag bases."""

    def __init__(
        self,
        echo_times_ms: np.ndarray | list[float],
        tr: float,
        regressor_extend: float = 15.0,
        knots_per_sec: float = 0.3,
        n_knots: int | None = None,
        knots: list[int] | None = None,
        basis_type: Literal["cr", "bs"] = "cr",
        event_duration: float | None = None,
        resample_tr: float = RESAMPLE_TR,
        slice_timing_ref: float = SLICE_TIMING_REF,
        ridge_alpha: float = 1.0,
        nuisance_ridge_alpha: float = 0.0,
        te_rescale_factor: float = 10.0,
        min_signal: float = 1e-6,
        max_t2star_ms: float = 500.0,
    ):
        super().__init__(
            echo_times_ms=echo_times_ms,
            ridge_alpha=ridge_alpha,
            nuisance_ridge_alpha=nuisance_ridge_alpha,
            te_rescale_factor=te_rescale_factor,
            min_signal=min_signal,
            max_t2star_ms=max_t2star_ms,
        )
        if tr <= 0:
            raise ValueError("tr must be positive")
        if regressor_extend < 0:
            raise ValueError("regressor_extend must be non-negative")
        if event_duration is not None and event_duration <= 0:
            raise ValueError("event_duration must be positive when provided")
        if resample_tr <= 0:
            raise ValueError("resample_tr must be positive")

        self.tr = float(tr)
        self.regressor_extend = float(regressor_extend)
        self.knots_per_sec = knots_per_sec
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type: Literal["cr", "bs"] = basis_type
        self.event_duration = event_duration
        self.resample_tr = float(resample_tr)
        self.slice_timing_ref = float(slice_timing_ref)

    def _clear_fit_results(self) -> None:
        super()._clear_fit_results()
        for attr in (
            "trial_bases_",
            "trial_types_",
            "trial_slices_",
            "trial_design_infos_",
            "trial_lag_max_",
            "resample_tr_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _clear_partial_fit_state(self) -> None:
        super()._clear_partial_fit_state()
        for attr in (
            "_partial_trial_bases",
            "_partial_trial_types",
            "_partial_trial_slices",
            "_partial_trial_design_infos",
            "_partial_trial_lag_max",
            "_partial_resample_tr",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _validate_event_df(self, event_df: pd.DataFrame) -> None:
        missing_columns = [
            column for column in EVENT_COLUMNS if column not in event_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns} in event dataframe")

    def _build_trial_metadata(
        self,
        trial_types: list[str],
        trial_bases: Mapping[str, SplineLagBasis],
    ) -> tuple[dict[str, slice], dict[str, Any]]:
        return _build_trial_slices(trial_types, trial_bases), {
            trial: trial_bases[trial].basis.design_info for trial in trial_types
        }

    def _initialize_prediction_metadata(
        self,
        trial_types: list[str],
        trial_bases: Mapping[str, SplineLagBasis],
        trial_lag_max: Mapping[str, float],
        resample_tr: float,
    ) -> None:
        self._partial_trial_types = list(trial_types)
        self._partial_trial_bases = dict(trial_bases)
        trial_slices, trial_design_infos = self._build_trial_metadata(
            self._partial_trial_types,
            self._partial_trial_bases,
        )
        self._partial_trial_slices = trial_slices
        self._partial_trial_design_infos = trial_design_infos
        self._partial_trial_lag_max = {
            name: float(value) for name, value in trial_lag_max.items()
        }
        self._partial_resample_tr = float(resample_tr)
        self._validate_prediction_metadata()

    def _validate_prediction_metadata(self) -> None:
        trial_type_keys = set(getattr(self, "_partial_trial_types", []))
        basis_keys = set(getattr(self, "_partial_trial_bases", {}))
        slice_keys = set(self._partial_trial_slices)
        info_keys = set(self._partial_trial_design_infos)
        max_keys = set(self._partial_trial_lag_max)
        if not trial_type_keys:
            raise ValueError("At least one trial type is required")
        if (
            trial_type_keys != basis_keys
            or trial_type_keys != slice_keys
            or trial_type_keys != info_keys
            or trial_type_keys != max_keys
        ):
            raise ValueError(
                "trial_types, trial_bases, trial_slices, trial_design_infos, and trial_lag_max must have the same keys"
            )
        if self._partial_resample_tr <= 0:
            raise ValueError("resample_tr must be positive")
        for trial, lag_max in self._partial_trial_lag_max.items():
            if lag_max < 0:
                raise ValueError(f"trial_lag_max[{trial!r}] must be non-negative")

    def _project_event_run(self, event_df: pd.DataFrame, n_vols: int) -> np.ndarray:
        event_reg, frametimes, h_frametimes = boxcar(
            event_df,
            tr=self.tr,
            resample_tr=self.resample_tr,
            n_vols=n_vols,
            slicetime_ref=self.slice_timing_ref,
            trial_types=self._partial_trial_types,
            impulse_dur=0.5,
        )
        event_regs_trial: list[np.ndarray] = []
        for trial_index, trial in enumerate(self._partial_trial_types):
            event_reg_proj = self._partial_trial_bases[trial].project(
                event_reg[trial_index],
                fill_val=0.0,
            )
            interp_func = interp1d(h_frametimes, event_reg_proj.T, kind="cubic")
            event_regs_trial.append(interp_func(frametimes).T)
        return normalize_run_regressors(np.hstack(event_regs_trial))

    def _validate_event_run_compatibility(self, event_df: pd.DataFrame) -> None:
        current_trials = {
            str(trial) for trial in event_df["trial_type"].unique().tolist()
        }
        known_trials = set(self._partial_trial_types)
        unexpected_trials = current_trials - known_trials
        if unexpected_trials:
            raise ValueError(
                "fit_partial encountered trial types that were not present when the event basis was initialized: "
                f"{sorted(unexpected_trials)}"
            )
        for trial in self._partial_trial_types:
            trial_durations = event_df.loc[
                event_df["trial_type"] == trial, "duration"
            ].to_numpy()
            if trial_durations.size == 0:
                continue
            if self.event_duration is not None:
                required_lag_max = self.event_duration + self.regressor_extend
            else:
                required_lag_max = float(
                    np.max(trial_durations) + self.regressor_extend
                )
            if required_lag_max > self._partial_trial_lag_max[trial] + 1e-9:
                raise ValueError(
                    "fit_partial encountered a longer event duration than the initialized basis supports. "
                    "Use fit() across all runs or set event_duration to a fixed value."
                )

    def _fit_projected_run(
        self,
        run_design_array: np.ndarray,
        run_data_array: np.ndarray,
        confounds: np.ndarray | None,
        initialize_metadata: bool = False,
        trial_types: list[str] | None = None,
        trial_bases: Mapping[str, SplineLagBasis] | None = None,
        trial_lag_max: Mapping[str, float] | None = None,
    ) -> None:
        self._validate_run_arrays(run_design_array, run_data_array)

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
                    "No confounds were provided; low-frequency trends are not removed from the monoexponential TE-dependent event model.",
                    stacklevel=2,
                )
            self._initialize_partial_fit_state(
                n_voxels=run_data_array.shape[0],
                n_regressors=run_design_array.shape[1],
                shared_confound_count=shared_confound_count,
            )
            if (
                not initialize_metadata
                or trial_types is None
                or trial_bases is None
                or trial_lag_max is None
            ):
                raise ValueError(
                    "Event prediction metadata must be initialized on the first run"
                )
            self._initialize_prediction_metadata(
                trial_types,
                trial_bases,
                trial_lag_max,
                self.resample_tr,
            )
        else:
            self._validate_existing_partial_dimensions(
                run_design_array,
                run_data_array,
                shared_confound_count,
            )

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

        self._accumulate_run_statistics(
            run_design_array,
            run_data_array,
            run_nuisance_design,
            nuisance_column_indices,
        )

    def _finalize_prediction_metadata(self) -> None:
        self.trial_bases_ = dict(getattr(self, "_partial_trial_bases", {}))
        self.trial_types_ = list(getattr(self, "_partial_trial_types", []))
        self.trial_slices_ = dict(getattr(self, "_partial_trial_slices", {}))
        self.trial_design_infos_ = dict(
            getattr(self, "_partial_trial_design_infos", {})
        )
        self.trial_lag_max_ = dict(getattr(self, "_partial_trial_lag_max", {}))
        self.resample_tr_ = float(
            getattr(self, "_partial_resample_tr", self.resample_tr)
        )

    def _build_prediction_time_grid(
        self,
        trial: str,
        lags: np.ndarray | list[float] | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = PREDICT_T_DELTA,
    ) -> tuple[np.ndarray, float | None, int | None]:
        if trial not in self.trial_slices_:
            raise ValueError(f"trial must be one of {list(self.trial_slices_)}")

        if lags is not None:
            pred_lags = np.asarray(lags, dtype=float).reshape(-1)
            if pred_lags.size == 0:
                raise ValueError("lags must contain at least one value")
            if np.any(~np.isfinite(pred_lags)):
                raise ValueError("lags must be finite")
            if np.any(pred_lags < 0):
                raise ValueError("Event-lag predictions must be non-negative")
            return pred_lags, None, int(pred_lags.size)

        lag_max_value = (
            self.trial_lag_max_[trial] if lag_max is None else float(lag_max)
        )
        if lag_max_value < 0:
            raise ValueError("lag_max must be non-negative")
        if n_eval is not None and n_eval < 1:
            raise ValueError("n_eval must be at least 1 when provided")
        if eval_delta <= 0:
            raise ValueError("eval_delta must be positive")

        if n_eval is None:
            pred_lags = np.arange(0.0, lag_max_value + 1e-9, eval_delta, dtype=float)
            pred_lags = pred_lags[pred_lags <= lag_max_value + 1e-9]
            if pred_lags.size == 0:
                pred_lags = np.asarray([0.0], dtype=float)
            return pred_lags, float(eval_delta), None

        pred_lags = np.linspace(0.0, lag_max_value, int(n_eval), dtype=float)
        return pred_lags, None, int(n_eval)

    def _build_trial_rows(
        self,
        trial: str,
        pred_lags: np.ndarray,
    ) -> np.ndarray:
        if trial not in self.trial_slices_:
            raise ValueError(f"trial must be one of {list(self.trial_slices_)}")
        if np.any(pred_lags < 0):
            raise ValueError("Event-lag predictions must be non-negative")

        pred_lag_steps = pred_lags / self.resample_tr_
        basis_rows = np.asarray(
            dmatrix(
                self.trial_design_infos_[trial],
                {"x": pred_lag_steps.reshape(-1, 1)},
            )
        )
        effect_rows = np.zeros((pred_lags.size, self.n_regressors_), dtype=float)
        trial_slice = self.trial_slices_[trial]
        if basis_rows.shape[1] != effect_rows[:, trial_slice].shape[1]:
            raise ValueError(
                f"Lag basis column count for {trial!r} does not match its registered slice."
            )
        effect_rows[:, trial_slice] = basis_rows
        return effect_rows

    def fit(
        self,
        event_dfs: list[pd.DataFrame],
        data: list[np.ndarray],
        confounds: list[np.ndarray] | None = None,
    ) -> "DistributedLagEventMonoexponentialEchoModel":
        if len(event_dfs) == 0 or len(data) == 0:
            raise ValueError("event_dfs and data must be non-empty lists")
        if len(event_dfs) != len(data):
            raise ValueError("event_dfs and data must have the same number of runs")
        if confounds is not None and len(confounds) != len(data):
            raise ValueError("confounds and data must have the same number of runs")
        for event_df in event_dfs:
            self._validate_event_df(event_df)

        trial_types = _get_ordered_trial_types(event_dfs)
        if not trial_types:
            raise ValueError("At least one trial type is required")

        outcome_data = [
            np.empty((run_data.shape[2], 1), dtype=float) for run_data in data
        ]
        event_regs, _, event_bases, _, trial_durations_extend = create_spline_event_reg(
            event_dfs=event_dfs,
            outcome_data=outcome_data,
            trial_types=trial_types,
            tr=self.tr,
            resample_tr=self.resample_tr,
            slice_timing_ref=self.slice_timing_ref,
            knots_per_sec=self.knots_per_sec,
            n_knots=self.n_knots,
            basis_type=self.basis_type,
            knots=self.knots,
            regressor_extend=self.regressor_extend,
            event_duration=self.event_duration,
        )

        confounds_list = [None] * len(data) if confounds is None else list(confounds)
        self._clear_fit_results()
        self._clear_partial_fit_state()
        if confounds is None:
            warnings.warn(
                "No confounds were provided; low-frequency trends are not removed from the monoexponential TE-dependent event model.",
                stacklevel=2,
            )

        for run_index, (run_design, run_data, run_confounds) in enumerate(
            zip(event_regs, data, confounds_list)
        ):
            self._fit_projected_run(
                run_design_array=np.asarray(run_design, dtype=float),
                run_data_array=np.asarray(run_data, dtype=float),
                confounds=run_confounds,
                initialize_metadata=run_index == 0,
                trial_types=trial_types if run_index == 0 else None,
                trial_bases=event_bases if run_index == 0 else None,
                trial_lag_max=trial_durations_extend if run_index == 0 else None,
            )

        return self.finalize_fit()

    def fit_partial(
        self,
        event_df: pd.DataFrame,
        data: np.ndarray,
        confounds: np.ndarray | None = None,
    ) -> "DistributedLagEventMonoexponentialEchoModel":
        self._validate_event_df(event_df)
        run_data_array = np.asarray(data, dtype=float)
        if not hasattr(self, "_partial_run_count"):
            trial_types = _get_ordered_trial_types([event_df])
            if not trial_types:
                raise ValueError("At least one trial type is required")
            event_regs, _, event_bases, _, trial_durations_extend = (
                create_spline_event_reg(
                    event_dfs=[event_df],
                    outcome_data=[np.empty((run_data_array.shape[2], 1), dtype=float)],
                    trial_types=trial_types,
                    tr=self.tr,
                    resample_tr=self.resample_tr,
                    slice_timing_ref=self.slice_timing_ref,
                    knots_per_sec=self.knots_per_sec,
                    n_knots=self.n_knots,
                    basis_type=self.basis_type,
                    knots=self.knots,
                    regressor_extend=self.regressor_extend,
                    event_duration=self.event_duration,
                )
            )
            run_design_array = np.asarray(event_regs[0], dtype=float)
            self._fit_projected_run(
                run_design_array=run_design_array,
                run_data_array=run_data_array,
                confounds=confounds,
                initialize_metadata=True,
                trial_types=trial_types,
                trial_bases=event_bases,
                trial_lag_max=trial_durations_extend,
            )
        else:
            self._validate_event_run_compatibility(event_df)
            run_design_array = np.asarray(
                self._project_event_run(event_df, n_vols=run_data_array.shape[2]),
                dtype=float,
            )
            self._fit_projected_run(
                run_design_array=run_design_array,
                run_data_array=run_data_array,
                confounds=confounds,
            )
        return self

    def predict_curve_across_lags(
        self,
        trial: str,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        lags: np.ndarray | list[float] | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = PREDICT_T_DELTA,
    ) -> MonoexponentialCurveAcrossLagsPrediction:
        pred_lags, eval_delta_out, n_eval_out = self._build_prediction_time_grid(
            trial=trial,
            lags=lags,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_trial_rows(trial, pred_lags)
        return self._predict_curve_from_rows(
            regressor_name=trial,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta_out,
            n_eval=n_eval_out,
        )

    def predict_curve_across_lags_for_echo(
        self,
        trial: str,
        echo_index: int,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        lags: np.ndarray | list[float] | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = PREDICT_T_DELTA,
    ) -> MonoexponentialCurveAcrossLagsForEchoPrediction:
        if echo_index < 0 or echo_index >= self.E:
            raise ValueError(f"echo_index must be between 0 and {self.E - 1}")

        across_lags = self.predict_curve_across_lags(
            trial=trial,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=lags,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        return MonoexponentialCurveAcrossLagsForEchoPrediction(
            pred_lags=across_lags.pred_lags.copy(),
            pred_curve=across_lags.pred_curve[:, echo_index, :],
            pred_log_curve=across_lags.pred_log_curve[:, echo_index, :],
            pred_log_effect=across_lags.pred_log_effect[:, echo_index, :],
            pred_log_nuisance=across_lags.pred_log_nuisance[:, echo_index, :],
            params=MonoexponentialCurveAcrossLagsForEchoPredictionParams(
                echo_index=echo_index,
                echo_time_ms=float(self.echo_times_ms[echo_index]),
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=trial,
                lag_min=across_lags.params.lag_min,
                lag_max=across_lags.params.lag_max,
                n_eval=across_lags.params.n_eval,
                eval_delta=across_lags.params.eval_delta,
            ),
        )

    def predict_curve_at_lag(
        self,
        trial: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialCurvePrediction:
        across_lags = self.predict_curve_across_lags(
            trial=trial,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return MonoexponentialCurvePrediction(
            echo_times_ms=across_lags.echo_times_ms.copy(),
            pred_curve=across_lags.pred_curve[0],
            pred_log_curve=across_lags.pred_log_curve[0],
            pred_log_effect=across_lags.pred_log_effect[0],
            pred_log_nuisance=across_lags.pred_log_nuisance[0],
            params=MonoexponentialCurvePredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                regressor=trial,
                lag=float(lag),
            ),
        )

    def predict_loglinear_params_across_lags(
        self,
        trial: str,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        lags: np.ndarray | list[float] | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = PREDICT_T_DELTA,
    ) -> MonoexponentialParamAcrossLagsPrediction:
        pred_lags, eval_delta_out, n_eval_out = self._build_prediction_time_grid(
            trial=trial,
            lags=lags,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_trial_rows(trial, pred_lags)
        return self._predict_loglinear_from_rows(
            regressor_name=trial,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta_out,
            n_eval=n_eval_out,
        )

    def predict_loglinear_params_at_lag(
        self,
        trial: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialParamPrediction:
        across_lags = self.predict_loglinear_params_across_lags(
            trial=trial,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return MonoexponentialParamPrediction(
            param_names=across_lags.param_names,
            pred_params=across_lags.pred_params[0],
            pred_effect_params=across_lags.pred_effect_params[0],
            pred_nuisance_params=across_lags.pred_nuisance_params[0],
            params=MonoexponentialParamPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="loglinear",
                regressor=trial,
                lag=float(lag),
            ),
        )

    def predict_t2star_across_lags(
        self,
        trial: str,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
        lags: np.ndarray | list[float] | None = None,
        lag_max: float | None = None,
        n_eval: int | None = None,
        eval_delta: float = PREDICT_T_DELTA,
    ) -> MonoexponentialDerivedParamAcrossLagsPrediction:
        pred_lags, eval_delta_out, n_eval_out = self._build_prediction_time_grid(
            trial=trial,
            lags=lags,
            lag_max=lag_max,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
        effect_rows = self._build_trial_rows(trial, pred_lags)
        return self._predict_t2star_from_rows(
            regressor_name=trial,
            pred_lags=pred_lags,
            effect_rows=effect_rows,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            eval_delta=eval_delta_out,
            n_eval=n_eval_out,
        )

    def predict_t2star_at_lag(
        self,
        trial: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialDerivedParamPrediction:
        across_lags = self.predict_t2star_across_lags(
            trial=trial,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lags=np.asarray([lag], dtype=float),
        )
        return MonoexponentialDerivedParamPrediction(
            param_names=across_lags.param_names,
            pred_params=across_lags.pred_params[0],
            params=MonoexponentialParamPredictionParams(
                pred_val=pred_val,
                run=run,
                include_intercept=include_intercept,
                parameterization="t2star",
                regressor=trial,
                lag=float(lag),
            ),
        )

    def evaluate(
        self,
        trial: str,
        eval_delta: float = PREDICT_T_DELTA,
        pred_val: float = 1.0,
        n_eval: int | None = None,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> MonoexponentialCurveAcrossLagsPrediction:
        return self.predict_curve_across_lags(
            trial=trial,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
            lag_max=None,
            n_eval=n_eval,
            eval_delta=eval_delta,
        )
