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
        regressor_slices : Mapping[str, slice] | None
                Optional mapping from regressor names to column slices in the design
                matrix. This is used by ``predict_curve_at_lag``.
        lag_design_infos : Mapping[str, Any] | None
                Optional Patsy ``design_info`` objects keyed by regressor name. These are
                used to evaluate the lag basis for prediction at arbitrary lag values.
        """
        if len(regressors) == 0 or len(data) == 0:
            raise ValueError("regressors and data must be non-empty lists")
        if len(regressors) != len(data):
            raise ValueError("regressors and data must have the same number of runs")

        if confounds is None:
            warnings.warn(
                "No confounds were provided; low-frequency trends are not removed from the TE-dependent model.",
                stacklevel=2,
            )
            confounds_list: list[np.ndarray | None] = [None] * len(data)
        else:
            if len(confounds) != len(data):
                raise ValueError("confounds and data must have the same number of runs")
            confounds_list = [
                None
                if run_confounds.size == 0
                else np.asarray(run_confounds, dtype=float)
                for run_confounds in confounds
            ]

        n_voxels: int | None = None
        n_regressors: int | None = None
        total_timepoints = 0
        total_nuisance = 0

        validated_designs: list[np.ndarray] = []
        validated_data: list[np.ndarray] = []
        for run_design, run_data in zip(regressors, data):
            run_design_array = np.asarray(run_design, dtype=float)
            run_data_array = np.asarray(run_data, dtype=float)
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
                    "Each run design must have the same number of timepoints as the "
                    "corresponding data tensor"
                )
            if n_voxels is None:
                n_voxels = run_data_array.shape[0]
            elif run_data_array.shape[0] != n_voxels:
                raise ValueError("All data tensors must have the same voxel dimension")
            if n_regressors is None:
                n_regressors = run_design_array.shape[1]
            elif run_design_array.shape[1] != n_regressors:
                raise ValueError(
                    "All regressor designs must have the same number of columns"
                )
            if np.any(~np.isfinite(run_design_array)):
                raise ValueError("Regressor designs must be finite")
            if np.any(~np.isfinite(run_data_array)):
                raise ValueError("Data tensors must be finite")
            validated_designs.append(run_design_array)
            validated_data.append(run_data_array)
            total_timepoints += run_design_array.shape[0]

        assert n_voxels is not None
        assert n_regressors is not None

        for run_design, run_confounds in zip(validated_designs, confounds_list):
            confound_cols = 0 if run_confounds is None else run_confounds.shape[1]
            if run_confounds is not None:
                if run_confounds.ndim != 2:
                    raise ValueError("Each confound array must be 2D (time x nuisance)")
                if run_confounds.shape[0] != run_design.shape[0]:
                    raise ValueError(
                        "Each confound matrix must have the same number of rows as the corresponding run"
                    )
                if np.any(~np.isfinite(run_confounds)):
                    raise ValueError("Confound matrices must be finite")
            total_nuisance += 1 + confound_cols

        X_concat = np.vstack(validated_designs)
        Y_concat = np.concatenate(
            [np.transpose(run_data, (2, 1, 0)) for run_data in validated_data],
            axis=0,
        )

        C_concat = np.zeros((total_timepoints, total_nuisance), dtype=float)
        intercept_column_indices: list[int] = []
        row_start = 0
        col_start = 0
        for run_design, run_confounds in zip(validated_designs, confounds_list):
            n_timepoints = run_design.shape[0]
            confound_cols = 0 if run_confounds is None else run_confounds.shape[1]
            row_slice = slice(row_start, row_start + n_timepoints)
            C_concat[row_slice, col_start] = 1.0
            intercept_column_indices.append(col_start)
            if run_confounds is not None and confound_cols > 0:
                C_concat[row_slice, col_start + 1 : col_start + 1 + confound_cols] = (
                    run_confounds
                )
            row_start += n_timepoints
            col_start += 1 + confound_cols

        XtX = X_concat.T @ X_concat
        XtC = X_concat.T @ C_concat
        CtC = C_concat.T @ C_concat
        XtY = np.einsum("tk,tev->kev", X_concat, Y_concat)
        CtY = np.einsum("tj,tev->jev", C_concat, Y_concat)

        effect_normal = kron(csc_matrix(XtX), eye(self.E, format="csc"), format="csc")
        cross_normal = kron(csc_matrix(XtC), eye(self.E, format="csc"), format="csc")
        nuisance_normal = kron(csc_matrix(CtC), eye(self.E, format="csc"), format="csc")

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
                XtY.reshape(effect_param_count, n_voxels),
                CtY.reshape(nuisance_param_count, n_voxels),
            ]
        )
        self._solver = QuadraticPenaltySolver(
            as_csc(full_normal),
            as_csc(full_penalty),
        )
        beta = self._solver.solve(rhs)

        effect_beta = beta[:effect_param_count].T.reshape(
            n_voxels, n_regressors, self.E
        )
        nuisance_beta = beta[effect_param_count:].T.reshape(
            n_voxels, total_nuisance, self.E
        )

        self.effect_coefs_ = effect_beta
        self.nuisance_coefs_ = nuisance_beta
        self.n_regressors_ = n_regressors
        self.n_nuisance_ = total_nuisance
        self.n_voxels_ = n_voxels
        self.intercept_column_indices_ = intercept_column_indices
        self.run_regressor_designs_ = [design.copy() for design in validated_designs]
        self.run_nuisance_designs_ = []
        col_start = 0
        for run_design, run_confounds in zip(validated_designs, confounds_list):
            n_timepoints = run_design.shape[0]
            confound_cols = 0 if run_confounds is None else run_confounds.shape[1]
            run_nuisance = np.zeros((n_timepoints, 1 + confound_cols), dtype=float)
            run_nuisance[:, 0] = 1.0
            if run_confounds is not None and confound_cols > 0:
                run_nuisance[:, 1:] = run_confounds
            self.run_nuisance_designs_.append(run_nuisance)
            col_start += 1 + confound_cols
        self.regressor_slices_ = dict(regressor_slices or {})
        self.lag_design_infos_ = dict(lag_design_infos or {})
        return self

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

    def predict_curve_at_lag(
        self,
        regressor: str,
        lag: float,
        pred_val: float = 1.0,
        run: int = 0,
        include_intercept: bool = True,
        nuisance_values: np.ndarray | None = None,
    ) -> EchoCurvePrediction:
        """Predict the echo dependence for one named regressor at a chosen lag."""
        if regressor not in self.regressor_slices_:
            raise ValueError(
                f"No regressor slice found for {regressor!r}. Provide regressor_slices in fit()."
            )
        if regressor not in self.lag_design_infos_:
            raise ValueError(
                f"No lag design info found for {regressor!r}. Provide lag_design_infos in fit()."
            )

        basis_row = np.asarray(
            dmatrix(
                self.lag_design_infos_[regressor],
                {"x": np.asarray([[lag]], dtype=float)},
            )
        ).reshape(-1)
        effect_row = np.zeros(self.n_regressors_, dtype=float)
        effect_row[self.regressor_slices_[regressor]] = basis_row
        prediction = self.predict_curve(
            regressor_values=effect_row,
            pred_val=pred_val,
            run=run,
            include_intercept=include_intercept,
            nuisance_values=nuisance_values,
        )
        prediction.params.regressor = regressor
        prediction.params.lag = float(lag)
        return prediction

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
        effect_coefs = self.effect_coefs_[:, :, echo_index]
        pred_effect = np.einsum("tk,vk->tv", run_effect_design, effect_coefs)

        nuisance_start = self.intercept_column_indices_[run]
        nuisance_stop = (
            self.intercept_column_indices_[run + 1]
            if run + 1 < len(self.intercept_column_indices_)
            else self.n_nuisance_
        )
        nuisance_coefs = self.nuisance_coefs_[
            :, nuisance_start:nuisance_stop, echo_index
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
