"""
Distributed lag modeling of physio signals, events, and fMRI data
"""

from dataclasses import dataclass
from typing import List, Literal

import warnings

import numpy as np
import pandas as pd

from patsy import dmatrix  # type: ignore
from scipy.stats import zscore
from sklearn.linear_model import Ridge

from task_arousal.constants import SLICE_TIMING_REF, EVENT_COLUMNS
from task_arousal.analysis.basis import create_spline_event_reg, SplineLagBasis
from task_arousal.analysis.utils import get_trials_from_event_dfs

# define the resampling of the event time course for boxcar function (in seconds)
RESAMPLE_TR = 0.01  # seconds

# define the the distance (in seconds) between samples of the predicted functional time course after event offset
PREDICT_T_DELTA = 1  # seconds


# dataclass for storing DLM prediction results
@dataclass
class DLMParams:
    lag_max: float
    lag_min: float
    n_eval: int | None
    pred_val: float
    pred_lags_step: np.ndarray
    pred_lags_sec: np.ndarray
    basis_type: str


@dataclass
class DLMPredResults:
    tr: float
    pred_outcome: np.ndarray
    dlm_params: DLMParams
    trial: str | None = None


class DistributedLagPhysioModel:
    """
    Distributed lag model of physio signals regressed onto functional
    MRI signals at each voxel (mass-univariate). Specifically, lags of
    the physio signal are projected on a B-spline basis and regressed onto
    functional MRI signals.

    Attributes
    ----------
    tr: float
        repetition time of the fMRI data (in seconds)
    nlags: int
        number of lags (shifts) of the physio signal in the forward direction
    nlags_neg: int
        number of lags (shifts) of the physio signal in the negative direction.
        Must be a negative integer. This allows modeling the association between
        functional and physio signals where the functional leads the physio signal.
    knots_per_sec: float
        number of knots per second in the spline basis across temporal lags. This ensures
        that varying duration trials have similar temporal smoothness in the basis. For example,
        a value of 0.5 results in one knot every 2 seconds. Default is 0.3 knots per second.
    n_knots: int | None
        fix the number of knots in the spline basis across temporal lags. If this parameter is set,
        the knots_per_sec parameter is ignored. Default is None.
    knots: List[int] | None
        knot locations for the spline basis across temporal lags. If supplied, this
        overrides the n_knots parameter. If this parameter is set, the knots_per_sec parameter and
        n_knots parameter are ignored.
    basis_type: Literal['cr','bs']
        basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        regress lags of physio signal onto voxel-wise functional time courses.

    predict()

    """

    def __init__(
        self,
        tr: float,
        nlags: int,
        neg_nlags: int = 0,
        knots_per_sec: float = 0.3,
        n_knots: int | None = None,
        knots: List[int] | None = None,
        basis_type: Literal["cr", "bs"] = "cr",
    ):
        # specify array of lags
        self.tr = tr
        self.nlags = nlags
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        self.neg_nlags = neg_nlags
        self.knots_per_sec = knots_per_sec
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis_type

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        fit regression model of physio lag spline basis regressed on functional
        time courses

        Parameters
        ----------
        X: np.ndarray
            The physio time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        Y: np.ndarray
            functional MRI time courses represented in an ndarray with time
            points along the rows and vertices in the columns (# of time
            points, # of vertices).

        Returns
        -------
        self: object
            Fitted model instance.
        """
        # check that X and Y have same number of time points
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of time points")
        # check that X has a single column
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("X must have a single column")
        # create B-spline basis across lags of physio signal
        self.basis = SplineLagBasis(
            knots_per_sec=self.knots_per_sec,
            tr=self.tr,
            n_knots=self.n_knots,
            knots=self.knots,
            basis_type=self.basis_type,  # type: ignore
        )
        self.basis.create(self.nlags, self.neg_nlags)
        # project physio signal lags on B-spline basis
        x_basis = self.basis.project(X)
        # create nan mask for x_basis
        self.nan_mask = np.isnan(x_basis).any(axis=1)

        # fit Linear regression model
        self.glm = Ridge(fit_intercept=False, alpha=10000)
        self.glm.fit(
            # normalize X basis
            np.array(zscore(x_basis[~self.nan_mask])),
            Y[~self.nan_mask],
        )
        return self

    def evaluate(
        self,
        lag_max: float | None = None,
        lag_min: float | None = None,
        n_eval: int | None = None,
        pred_val: float = 1.0,
        eval_delta: float = PREDICT_T_DELTA,
    ) -> DLMPredResults:
        """
        Evaluate the model at user-specified lags and values of the physio signal.

        Parameters
        ----------
        lag_max: float
            The length of lags of the physio signal to predict functional time
            courses for. If None, set to nlag specified in initialization. (
            default: None)
        lag_min: float
            The minimium lag of the physio signal to predict functional time
            courses for. Must be a negative integer. If None, set to neg_nlag
            specified in initialization. (default: None)
        n_eval: int | None
            Fix the number of samples to predict between lag_min and lag_max.
            If specified, eval_delta is ignored. If None, eval_delta is used to
            determine the number of samples.
        eval_delta: float
            The time (in seconds) between samples to predict functional time.
            Ignored if n_eval is specified. (default: PREDICT_T_DELTA)
        pred_val: float
            The predicted physio signal value used to predict functional time
            courses (default: 1.0).

        Returns
        -------
        dlm_pred: DistributedLagModelPredResults
            Container object for distribued lag model prediction results
        """
        # if lag_max is None, set nlags
        if lag_max is None:
            lag_max = self.nlags
        # if lag_min is None, set neg_nlags
        if lag_min is None:
            lag_min = self.neg_nlags
        else:
            if lag_min > 0:
                raise ValueError("lag_min must be a negative integer")

        # Convert lag bounds (steps) to seconds using TR
        lag_min_sec = float(lag_min) * self.tr
        lag_max_sec = float(lag_max) * self.tr
        # Build prediction timeline in seconds with fixed delta or fixed count
        if n_eval is None:
            # Constant spacing at eval_delta
            pred_lags_sec = np.arange(lag_min_sec, lag_max_sec + 1e-9, eval_delta)
            # Clip any overshoot
            pred_lags_sec = pred_lags_sec[pred_lags_sec <= lag_max_sec]
        else:
            # Fixed number of samples from min to max
            pred_lags_sec = np.linspace(lag_min_sec, lag_max_sec, n_eval)

        # Map seconds to lag-step units (basis built at TR step)
        pred_lags_step = pred_lags_sec / self.tr
        # project lag vector (in steps) onto B-spline basis
        pred_basis = dmatrix(
            self.basis.basis.design_info, {"x": pred_lags_step.reshape(-1, 1)}
        )
        # project prediction value on lag B-spline basis
        physio_pred = [
            pred_val * pred_basis[:, spline_n]
            for spline_n in range(pred_basis.shape[1])
        ]
        physio_pred = np.vstack(physio_pred).T
        # Get predictions from model
        pred_func = self.glm.predict(physio_pred)
        # package output in container object
        dlm_pred = DLMPredResults(
            tr=self.tr,
            pred_outcome=pred_func,
            dlm_params=DLMParams(
                lag_max=lag_max_sec,
                lag_min=lag_min_sec,
                n_eval=None if n_eval is None else int(n_eval),
                pred_val=pred_val,
                pred_lags_step=pred_lags_step,
                pred_lags_sec=pred_lags_sec,
                basis_type=self.basis_type,
            ),
        )
        return dlm_pred


class DistributedLagEventModel:
    """
    Distributed lag model of task events regressed onto 1) functional
    MRI signals at each voxel (mass-univariate) or 2) physio signals. Specifically, lags of
    the task event signal are projected on a B-spline basis and regressed onto
    functional MRI signals or physio signals.

    Attributes
    ----------
    tr: float
        repetition time of the fMRI data (in seconds)
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

    """

    def __init__(
        self,
        tr: float,
        regressor_extend: float = 10.0,
        knots_per_sec: float = 0.3,
        n_knots: int | None = None,
        knots: List[int] | None = None,
        basis_type: Literal["cr", "bs"] = "cr",
        regressor_duration: float | None = None,
    ):
        self.tr = tr
        self.regressor_extend = regressor_extend
        self.knots_per_sec = knots_per_sec
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis_type
        self.regressor_duration = regressor_duration

    def fit(self, event_dfs: List[pd.DataFrame], outcome_data: List[np.ndarray]):
        """
        fit regression model of physio lag spline basis regressed on functional
        time courses or physio signals.

        Parameters
        ----------
        event_dfs: List[pd.DataFrame]
            List of dataframes containing event information (onset, duration, etc.).
        outcome_data: List[np.ndarray]
            List of fMRI 2D datasets (2D - time x voxels) or physio signals (2D - time x signals).
            This should be in the same order as event_dfs (i.e., outcome_data[i] corresponds to event_dfs[i]).
            This should not be concatenated data across runs/sessions.

        Returns
        -------
        self: object
            Fitted model instance.
        """
        # check that event_dfs and outcome_data have same length
        if len(event_dfs) != len(outcome_data):
            raise ValueError("event_dfs and outcome_data must have the same length")
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
            outcome_data=outcome_data,
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

        # concatenate outcome data across sessions/runs
        outcome_concat = np.vstack(outcome_data)
        # concatenate event regressors across sessions/runs
        event_regs_concat = np.vstack(self.event_regs)
        # zscore event regressors
        event_regs_concat = np.array(zscore(event_regs_concat, axis=0))
        # fit Ridge regression model
        self.glm = Ridge(fit_intercept=False, alpha=1.0)
        self.glm.fit(X=event_regs_concat, y=outcome_concat)
        return self

    def evaluate(
        self,
        trial: str,
        eval_delta: float = PREDICT_T_DELTA,
        pred_val: float = 1.0,
        n_eval: int | None = None,
    ) -> DLMPredResults:
        """
        Evaluate model predictions for a specific trial type.

        Parameters
        ----------
        trial: str
            The trial type to evaluate the model for. Must be one of the
            trial types used in the model fitting.
        eval_delta: float
            The time (in seconds) between samples to predict functional time. Ignored
            if n_eval is specified. (default: PREDICT_T_DELTA).
        pred_val: float
            The predicted event value used to predict functional time
            courses in z-score units (default: 1.0).
        n_eval: int | None
            Fix the number of samples to predict between the event onset and max_duration for
            the predicted functional time course or physio signal. If specified,
            eval_delta is ignored. If None, eval_delta is used to determine the number of samples.

        Returns
        -------
        dlm_pred: DistributedLagModelPredResults
            Container object for distribued lag model prediction results
        """
        # check that trial is in trial_types
        if trial not in self.trial_types:
            raise ValueError(f"trial must be one of {self.trial_types}")

        # Determine prediction sampling timeline in seconds
        # Use saved max regressor duration for this trial from fit()
        max_duration_sec = self.trial_durations_extend[trial]
        if n_eval is None:
            # Sample with fixed delta (eval_delta) from 0 to max_duration
            # np.arange ensures constant spacing equal to eval_delta
            pred_times_sec = np.arange(0.0, max_duration_sec + 1e-9, eval_delta)
            # Clip any tiny numerical overshoot to stay within duration
            pred_times_sec = pred_times_sec[pred_times_sec <= max_duration_sec]
        else:
            # Override with fixed number of samples from 0 to max_duration
            pred_times_sec = np.linspace(0.0, max_duration_sec, n_eval)

        # Convert seconds to the basis' lag-step units (built at RESAMPLE_TR)
        # Basis was created on integer lag indices measured at RESAMPLE_TR;
        # evaluating at arbitrary times requires mapping t_sec -> t_steps.
        pred_lags_steps = pred_times_sec / RESAMPLE_TR

        # Project lag positions onto the fitted spline design (same columns as training)
        pred_basis = dmatrix(
            self.basis[trial].basis.design_info, {"x": pred_lags_steps.reshape(-1, 1)}
        )
        # project prediction value on lag B-spline basis
        event_pred = [
            pred_val * pred_basis[:, spline_n]
            for spline_n in range(pred_basis.shape[1])
        ]
        event_pred = np.vstack(event_pred).T
        # create full event regressor with zeros for other trial types
        n_total_regs = len(self.event_reg_cols)
        n_trial_regs = self.basis[trial].basis.shape[1]
        trial_start_idx = self.event_reg_cols.index(f"{trial}_lag_spline1")
        event_pred_full = np.zeros((event_pred.shape[0], n_total_regs))
        event_pred_full[:, trial_start_idx : trial_start_idx + n_trial_regs] = (
            event_pred
        )
        # Get predictions from model
        pred_func = self.glm.predict(event_pred_full)
        # package output in container object
        dlm_pred = DLMPredResults(
            tr=self.tr,
            pred_outcome=pred_func,
            trial=trial,
            dlm_params=DLMParams(
                lag_max=max_duration_sec,
                lag_min=0.0,
                n_eval=None if n_eval is None else int(n_eval),
                pred_val=pred_val,
                pred_lags_step=pred_lags_steps,
                pred_lags_sec=pred_times_sec,
                basis_type=self.basis_type,
            ),
        )
        return dlm_pred
