"""
Distributed lag modeling of physio signals, events, and fMRI data
"""
from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from patsy import dmatrix # type: ignore
from scipy.interpolate import interp1d
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression

from task_arousal.constants import TR, SLICE_TIMING_REF, EVENT_COLUMNS
from task_arousal.analysis.utils import lag_mat, boxcar

# define the resampling of the event time course for boxcar function (in seconds)
RESAMPLE_TR = 0.01 # seconds

# dataclass for storing DLM prediction results
@dataclass
class DLMParams:
    lag_max: float
    lag_min: float
    n_eval: int
    pred_val: float
    pred_lags: np.ndarray
    basis_type: str

@dataclass
class DLMPredResults:
    pred_outcome: np.ndarray
    dlm_params: DLMParams
    trial: str | None = None

# dataclass for storing DLM interaction model parameters
@dataclass
class DLMCAParams:
    event_lags: np.ndarray
    physio_lags: int
    regressor_duration: float
    n_knots_event: int
    n_knots_physio: int
    basis_type: str

@dataclass
class DLMCAResults:
    # store results of commonality analysis
    dlm_params: DLMCAParams
    r2_full: np.ndarray
    r2_common: np.ndarray
    r2_physio_unique: np.ndarray
    r2_event_unique: np.ndarray


class BSplineLagBasis:
    """
    Spline basis for modeling temporal lags of a physio signal based on
    scikit-learn fit/transform API. Specifically, a B-spline basis is fit
    along the columns of a lag matrix (rows: time courses; columns: lags),
    where the first column is the original time course, the second column
    is the original time course lagged by one time point, the third column
    lagged by two time points, out to N lags (specified by nlags parameter).
    You can also specify negative lags (specified by neg_nlags parameter).

    Attributes
    ----------
    n_knots: int
        number of knots in the spline basis across temporal lags. Controls
        the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    knots: List[int]
        Locations of knots in spline basis across temporal lags. If provided,
        the n_knots parameter is ignored.
    basis_type: Literal['ns','bs']
        basis type for the spline basis. 'ns' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        fit B-spline basis to lags of the signal.
    transform(X, y)
        project lags of the signal onto the B-spline basis. X is the physio
        time course represented in an ndarray with time points along the
        rows and a single column (# of time points, 1).

    """
    def __init__(
        self, 
        n_knots: int = 5,
        knots: List[int] | None = None,
        basis_type: Literal['cr','bs'] = 'bs'
    ):
        # specify knots parameters
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis_type

    def create(self, nlags: int, neg_nlags: int = 0) -> 'BSplineLagBasis':
        """
        create spline basis over lags of physio signal

        Parameters
        ----------
        nlags: int
            number of lags (shifts) of the signal in the forward direction
        neg_nlags: int
            number of lags (shifts) of the signal in the negative direction.
            Must be a negative integer. This allows modeling the association between
            functional and physio signals where the functional leads the physio signal.
        """
        # create spline basis from sklearn SplineTransformer
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        
        # specify array of lags
        self.lags = np.arange(neg_nlags, nlags+1)
        if self.knots is not None:
            self.basis = dmatrix(
                f'{self.basis_type}(x, knots=self.knots) - 1',
                {'x': self.lags}
            )
        else:
            self.basis = dmatrix(
                f'{self.basis_type}(x, df=self.n_knots) - 1',
                {'x': self.lags}
            )
    
        return self

    def project(
        self, 
        X: np.ndarray, 
        fill_val: float = np.nan
    ) -> np.ndarray:
        """
        project lags of physio signal onto spline basis

        Parameters
        ----------
        X: np.ndarray
            The physio time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        fill_val: float, optional
            Value to use for filling in missing values after shifting.
            Defaults to np.nan.

        Returns
        -------
        lag_proj: np.ndarray
            Physio signal projected on B-spline basis.
        """
        # create lag matrix
        lagmat = lag_mat(X, self.lags.tolist(), fill_val=fill_val)
        # get number of splines
        n_splines = self.basis.shape[1]
        # allocate memory
        lag_proj = np.empty(
            (lagmat.shape[0], n_splines),
            dtype=lagmat.dtype
        )
        for spline_n in np.arange(n_splines):
            lag_proj[:,spline_n] = np.dot(lagmat, self.basis[:,spline_n])

        return lag_proj


class DistributedLagPhysioModel:
    """
    Distributed lag model of physio signals regressed onto functional
    MRI signals at each voxel (mass-univariate). Specifically, lags of
    the physio signal are projected on a B-spline basis and regressed onto
    functional MRI signals.

    Attributes
    ----------
    nlags: int
        number of lags (shifts) of the physio signal in the forward direction
    nlags_neg: int
        number of lags (shifts) of the physio signal in the negative direction.
        Must be a negative integer. This allows modeling the association between
        functional and physio signals where the functional leads the physio signal.
    n_knots: int
        number of knots in the spline basis across temporal lags. Controls
        the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    knots: List[int]
        knot locations for the spline basis across temporal lags. If supplied, this
        overrides the n_knots parameter.
    basis: Literal['cr','bs']
        basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        regress lags of physio signal onto voxel-wise functional time courses.

    predict()

    """
    def __init__(
        self,
        nlags: int,
        neg_nlags: int = 0,
        n_knots: int = 5,
        knots: List[int] | None = None,
        basis: Literal['cr','bs'] = 'cr'
    ):
        # specify array of lags
        self.nlags = nlags
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        self.neg_nlags = neg_nlags
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis

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
        self.basis = BSplineLagBasis(
            n_knots=self.n_knots, knots=self.knots, 
            basis_type=self.basis_type # type: ignore
        )
        self.basis.create(self.nlags, self.neg_nlags)
        # project physio signal lags on B-spline basis
        x_basis = self.basis.project(X)
        # create nan mask for x_basis
        self.nan_mask = np.isnan(x_basis).any(axis=1)

        # fit Linear regression model
        self.glm = Ridge(fit_intercept=False, alpha=1.0)
        self.glm.fit(
            # normalize X basis
            np.array(zscore(x_basis[~self.nan_mask])),
            Y[~self.nan_mask]
        )
        return self

    def evaluate(
        self,
        lag_max: float | None = None,
        lag_min: float | None = None,
        n_eval: int = 30,
        pred_val: float = 1.0
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
        n_eval: int
            Number of interpolated samples to predict functional time
            courses for between lag_min and lag_max.
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

        # specify lags for prediction (number of samples set by n_eval )
        pred_lags = np.linspace(lag_min, lag_max, n_eval)
        # project lag vector onto B-spline basis
        pred_basis = dmatrix(
            self.basis.basis.design_info,
            {'x': pred_lags.reshape(-1, 1)}
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
            pred_outcome = pred_func,
            dlm_params = DLMParams(
                lag_max = lag_max,
                lag_min = lag_min,
                n_eval = n_eval,
                pred_val = pred_val,
                pred_lags = pred_lags,
                basis_type = self.basis_type,
            )   
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
    regressor_duration: float
        duration of the spline regressors - i.e. the duration after onset of the event. This 
        should be set around the expected duration of the hemodynamic response
        to the event (default: 20.0 seconds).
    n_knots: int
        number of knots in the spline basis across temporal lags. Controls
        the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    knots: List[int]
        knot locations for the spline basis across temporal lags. If supplied, this
        overrides the n_knots parameter.
    basis: Literal['cr','bs']
        basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        regress lags of task events onto voxel-wise functional time courses or physio signals.

    predict()

    """
    def __init__(
        self,
        regressor_duration: float = 20.0,
        n_knots: int = 5,
        knots: List[int] | None = None,
        basis: Literal['cr','bs'] = 'cr'
    ):
        self.regressor_duration = regressor_duration
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis

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
        
        # calculate number of lags based on regressor duration and TR
        self.n_lags = int(np.ceil(self.regressor_duration / RESAMPLE_TR))
        # create spline basis
        self.basis = BSplineLagBasis(
            n_knots=self.n_knots,
            knots=self.knots,
            basis_type=self.basis_type # type: ignore
        )
        self.basis.create(self.n_lags, neg_nlags=0)

        # get trial types from first event df
        self.trial_types = event_dfs[0]['trial_type'].unique().tolist()
        # check that all event dfs have same trial types
        for i, df in enumerate(event_dfs):
            unique_trials = df['trial_type'].unique().tolist()
            if not all(trial in self.trial_types for trial in unique_trials):
                raise ValueError(f"Event dataframe {i} has different trial types than the first dataframe")
        
        # create column names for event regressors
        self.event_reg_cols = [
            f"{trial}_lag_spline{n+1}"
            for trial in self.trial_types
            for n in range(self.basis.basis.shape[1])
        ]
        # create event regressors for each session/run
        self.event_regs = []
        for i, (event_df, outcome_d) in enumerate(zip(event_dfs, outcome_data)):
            n_vols = outcome_d.shape[0]
            # create boxcar event regressor resampled at RESAMPLE_TR
            event_reg, frametimes, h_frametimes = boxcar(
                event_df,
                tr=TR,
                resample_tr=RESAMPLE_TR,
                n_vols=n_vols,
                slicetime_ref=SLICE_TIMING_REF,
                trial_types=self.trial_types,
                impulse_dur=0.5
            )
            # project each trial event regressor onto spline basis
            events_regs_trial = []
            for i, trial in enumerate(self.trial_types):
                # project event regressor onto spline basis
                # fill in NaNs with 0 to keep same length
                event_reg_proj = self.basis.project(event_reg[i], fill_val=0.0)
                # downsample (interpolate) event regressor to match fmri times
                interp_func = interp1d(
                    h_frametimes,
                    event_reg_proj.T,
                    kind='cubic'
                )
                event_reg_proj = interp_func(frametimes).T
                # trim fmri_img and event_reg to same length
                events_regs_trial.append(event_reg_proj)
            # create design matrix by concatenating trial event regressors
            event_regs_trial = np.hstack(events_regs_trial)
            self.event_regs.append(event_regs_trial)

        # concatenate outcome data across sessions/runs
        outcome_concat = np.vstack(outcome_data)
        # concatenate event regressors across sessions/runs
        event_regs_concat = np.vstack(self.event_regs)
        # zscore event regressors
        event_regs_concat = np.array(zscore(event_regs_concat, axis=0))

        # fit Ridge regression model
        self.glm = Ridge(fit_intercept=False, alpha=1.0)
        self.glm.fit(
            X=event_regs_concat,
            y=outcome_concat
        )
        return self

    def evaluate(
        self,
        trial: str,
        pred_val: float = 1.0,
        n_eval: int = 30,
    ) -> DLMPredResults:
        """
        Evaluate model predictions for a specific trial type.

        Parameters
        ----------
        trial: str
            The trial type to evaluate the model for. Must be one of the
            trial types used in the model fitting.
        n_eval: int
            Number of samples between the event onset and max_duration for 
            the predicted functional time course or physio signal.

        Returns
        -------
        dlm_pred: DistributedLagModelPredResults
            Container object for distribued lag model prediction results
        """
        # check that trial is in trial_types
        if trial not in self.trial_types:
            raise ValueError(f"trial must be one of {self.trial_types}")

        # specify lags for prediction (number of samples set by n_eval )
        pred_lags = np.linspace(0, self.n_lags, n_eval)
        # project lag vector onto B-spline basis
        pred_basis = dmatrix(
            self.basis.basis.design_info,
            {'x': pred_lags.reshape(-1, 1)}
        )
        # project prediction value on lag B-spline basis
        event_pred = [
            pred_val * pred_basis[:, spline_n]
            for spline_n in range(pred_basis.shape[1])
         ]
        event_pred = np.vstack(event_pred).T
        # create full event regressor with zeros for other trial types
        n_total_regs = len(self.event_reg_cols)
        n_trial_regs = self.basis.basis.shape[1]
        trial_start_idx = self.event_reg_cols.index(f"{trial}_lag_spline1")
        event_pred_full = np.zeros((event_pred.shape[0], n_total_regs))
        event_pred_full[:, trial_start_idx:trial_start_idx+n_trial_regs] = event_pred
        # Get predictions from model
        pred_func = self.glm.predict(event_pred_full)
        # package output in container object
        dlm_pred = DLMPredResults(
            pred_outcome = pred_func,
            trial = trial,
            dlm_params = DLMParams(
                lag_max = self.n_lags,
                lag_min = 0,
                n_eval = n_eval,
                pred_val = pred_val,
                pred_lags = pred_lags,
                basis_type = self.basis_type,
            )
        )
        return dlm_pred


class DistributedLagCommonalityAnalysis:
    """
    Partitioning of the unique and common variance in functional MRI time courses explained by
    task events and physio signals. Task events and physio signals and their lags are projected
    on a B-spline basis and regressed onto functional MRI signals at each voxel (mass-univariate).

    Attributes
    ----------
    physio_lag: int
        The chosen lag (in TRs) to include for the physiological regressor. Defaults to 5.
    regressor_duration: float
        duration of the spline regressors - i.e. the duration after onset of the event. This 
        should be set around the expected duration of the hemodynamic response
        to the event (default: 20.0 seconds).
    n_knots_event: int
        number of knots in the spline basis across temporal lags of the event regressors. 
        Controls the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    knots_events: List[int] | None
        knot locations for the spline basis across temporal lags of the event regressors. If supplied, this
        overrides the n_knots_event parameter.
    basis: Literal['cr','bs']
        basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        regress lags of task events onto voxel-wise functional time courses or physio signals.

    predict()

    """
    def __init__(
        self,
        physio_lags: int = 10,
        regressor_duration: float = 20.0,
        n_knots_event: int = 5,
        n_knots_physio: int = 5,
        physio_knots: List[int] | None = None,
        event_knots: List[int] | None = None,
        basis: Literal['cr','bs'] = 'cr'
    ):
        self.physio_lags = physio_lags
        self.regressor_duration = regressor_duration
        self.n_knots_event = n_knots_event
        self.n_knots_physio = n_knots_physio
        self.event_knots = event_knots
        self.physio_knots = physio_knots
        self.basis_type = basis

    def fit(
        self, 
        event_dfs: List[pd.DataFrame], 
        fmri_data: List[np.ndarray],
        physio_data: Dict[str, List[np.ndarray]]
    ) -> DLMCAResults:
        """
        fit regression model of combined event and physio lag spline basis
        regressed on functional time courses.

        Parameters
        ----------
        event_dfs: List[pd.DataFrame]
            List of dataframes containing event information (onset, duration, etc.).
        fmri_data: List[np.ndarray]
            List of fMRI 2D datasets (2D - time x voxels).
            This should be in the same order as event_dfs (i.e., fmri_data[i] corresponds to event_dfs[i]).
            This should not be concatenated data across runs/sessions.
        physio_data: Dict[str, List[np.ndarray]]
            Dictionary of physio signal (2D - time x 1) arrays, keyed by physio label.
            This should be in the same order as event_dfs (i.e., physio_data[physio_label][i] corresponds to event_dfs[i]).
            This should not be concatenated data across runs/sessions.

        Returns
        -------
        self: object
            Fitted model instance.
        """
        # check that event_dfs and fmri_data have same length
        if len(event_dfs) != len(fmri_data):
            raise ValueError("event_dfs and fmri_data must have the same length")
        # check that event_dfs and physio_data have same length
        if any(len(event_dfs) != len(physio_data[physio_label]) for physio_label in physio_data):
            raise ValueError("event_dfs and physio_data must have the same length")
        # check that physio_data have required shape
        for physio_label, physio in physio_data.items():
            for i, physio_subj in enumerate(physio):
                if physio_subj.ndim != 2 or physio_subj.shape[1] != 1:
                    raise ValueError(f"Invalid shape for {physio_label} for index {i}: {physio_subj.shape}")
        # check that event_dfs have required columns
        for i, df in enumerate(event_dfs):
            if not all(col in df.columns for col in EVENT_COLUMNS):
                raise ValueError(f"Missing columns: {EVENT_COLUMNS} in dataframe {i}")
        
        # calculate number of lags based on regressor duration and TR
        self.n_lags_event = int(np.ceil(self.regressor_duration / RESAMPLE_TR))
        # create spline basis for event regressors
        self.basis_event = BSplineLagBasis(
            n_knots=self.n_knots_event,
            knots=self.event_knots,
            basis_type=self.basis_type # type: ignore
        )
        self.basis_event.create(self.n_lags_event, neg_nlags=0)

        # get trial types from first event df
        self.trial_types = event_dfs[0]['trial_type'].unique().tolist()
        # check that all event dfs have same trial types
        for i, df in enumerate(event_dfs):
            unique_trials = df['trial_type'].unique().tolist()
            if not all(trial in self.trial_types for trial in unique_trials):
                raise ValueError(f"Event dataframe {i} has different trial types than the first dataframe")
        
        # create column names for event regressors
        self.event_reg_cols = [
            f"{trial}_lag_spline{n+1}"
            for trial in self.trial_types
            for n in range(self.basis_event.basis.shape[1])
        ]
        # create event regressors for each session/run
        self.event_regs = []
        for i, (event_df, outcome_d) in enumerate(zip(event_dfs, fmri_data)):
            n_vols = outcome_d.shape[0]
            # create boxcar event regressor resampled at RESAMPLE_TR
            event_reg, frametimes, h_frametimes = boxcar(
                event_df,
                tr=TR,
                resample_tr=RESAMPLE_TR,
                n_vols=n_vols,
                slicetime_ref=SLICE_TIMING_REF,
                trial_types=self.trial_types,
                impulse_dur=0.5
            )
            # project each trial event regressor onto spline basis
            events_regs_trial = []
            for i, trial in enumerate(self.trial_types):
                # project event regressor onto spline basis
                # fill in NaNs with 0 to keep same length
                event_reg_proj = self.basis_event.project(event_reg[i], fill_val=0.0)
                # downsample (interpolate) event regressor to match fmri times
                interp_func = interp1d(
                    h_frametimes,
                    event_reg_proj.T,
                    kind='cubic'
                )
                event_reg_proj = interp_func(frametimes).T
                # z-score event regressor - for comparability with physio regressor
                event_reg_proj = (
                    (event_reg_proj - np.mean(event_reg_proj, axis=0)) / np.std(event_reg_proj, axis=0)
                )
                # trim fmri_img and event_reg to same length
                events_regs_trial.append(event_reg_proj)
            # create design matrix by concatenating trial event regressors
            event_regs_trial = np.hstack(events_regs_trial)
            self.event_regs.append(event_regs_trial)
        
        # create B-spline basis across lags of physio signal
        self.basis_physio = BSplineLagBasis(
            n_knots=self.n_knots_physio, knots=self.physio_knots, 
            basis_type=self.basis_type # type: ignore
        )
        self.basis_physio.create(self.physio_lags, 0)
        # get physio labels
        self.physio_labels = list(physio_data.keys())
        # define physio regressor column names
        self.physio_reg_cols = []
        # loop through physio signals and project onto spline basis
        self.physio_regs = []
        for physio_label in self.physio_labels:
            physio_d = physio_data[physio_label]

            # extend physio regressor column names
            self.physio_reg_cols.extend([
                f"physio_{physio_label}_lag_spline{n+1}"
                for n in range(self.basis_physio.basis.shape[1])
            ])
            # create physio regressor for each session/run
            physio_regs_session = []
            for i, physio_d in enumerate(physio_data[physio_label]):
                if physio_d.shape[0] != fmri_data[i].shape[0]:
                    raise ValueError(
                        f"physio data {physio_label}, index {i} and fmri_data {i} must have the same"
                         " number of time points"
                    )
                # project physio signal lags on B-spline basis
                physio_reg_proj = self.basis_physio.project(physio_d, fill_val=0.0)
                physio_regs_session.append(physio_reg_proj)

            # conctatenate physio regressors across sessions/runs
            self.physio_regs.append(np.vstack(physio_regs_session))
        
        # concatenate physio regressors across physio signals
        physio_regs_concat = np.hstack(self.physio_regs)
        # concatenate fmri data across sessions/runs
        fmri_concat = np.vstack(fmri_data)
        # concatenate event regressors across sessions/runs
        event_regs_concat = np.vstack(self.event_regs)

        # concatenate event and physio regressors
        all_regs_concat = np.hstack([event_regs_concat, physio_regs_concat])

        # concatenate all colulmn labels
        self.all_reg_cols = self.event_reg_cols + self.physio_reg_cols

        # estimate total variance explained
        self.glm_full = LinearRegression(fit_intercept=False)
        self.glm_full.fit(
            X=all_regs_concat,
            y=fmri_concat
        )
        pred_fmri = self.glm_full.predict(
            X=all_regs_concat
        )
        self.r2_full = r2_score(
            y_true=fmri_concat,
            y_pred=pred_fmri,
            multioutput='raw_values'
        )

        # estimate variance explained by event regressors only
        self.glm_event = LinearRegression(fit_intercept=False)
        self.glm_event.fit(
            X=event_regs_concat,
            y=fmri_concat
        )
        fmri_pred = self.glm_event.predict(X=event_regs_concat)
        self.r2_event = r2_score(
            y_true=fmri_concat,
            y_pred=fmri_pred,
            multioutput='raw_values'
        )
        # estimate variance explained by physio regressors only
        self.glm_physio = LinearRegression(fit_intercept=False)
        self.glm_physio.fit(
            X=physio_regs_concat,
            y=fmri_concat
        )
        pred_fmri = self.glm_physio.predict(X=physio_regs_concat)
        self.r2_physio = r2_score(
            y_true=fmri_concat,
            y_pred=pred_fmri,
            multioutput='raw_values'
        )
        # estimate unique variance explained by event regressors
        self.r2_event_unique = (
            self.r2_full - self.r2_physio
        )
        # estimate unique variance explained by physio regressors
        self.r2_physio_unique = (
            self.r2_full - self.r2_event
        )
        # estimate the common variance explained by event and physio regressors
        self.r2_common = (
            self.r2_full - self.r2_event_unique - self.r2_physio_unique
        )
        return DLMCAResults(
            dlm_params = DLMCAParams(
                event_lags = np.arange(0, self.n_lags_event+1),
                physio_lags = self.physio_lags,
                regressor_duration = self.regressor_duration,
                n_knots_event = self.n_knots_event,
                n_knots_physio = self.n_knots_physio,
                basis_type = self.basis_type,
            ),
            r2_full = self.r2_full,
            r2_common = self.r2_common,
            r2_physio_unique = self.r2_physio_unique,
            r2_event_unique = self.r2_event_unique
        )
