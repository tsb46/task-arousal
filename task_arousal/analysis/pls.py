"""
Partial Least Squares regression analysis module.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from mbpls.mbpls import MBPLS
from scipy.stats import zscore

from task_arousal.constants import TR, SLICE_TIMING_REF, EVENT_COLUMNS
from task_arousal.analysis.dlm import SplineLagBasis
from task_arousal.analysis.utils import boxcar

# define the resampling of the event time course for boxcar function (in seconds)
RESAMPLE_TR = 0.01 # seconds

# dataclass for storing PLS prediction results
@dataclass
class PLSParams:
    n_components: int
    physio_lags: int
    regressor_duration: float
    n_knots_event: int
    n_knots_physio: int
    basis_type: str

@dataclass
class PLSResults:
    pls_params: PLSParams
    pls: MBPLS
    physio_col_labels: List[str]
    event_col_labels: List[str]
    trial_types: List[str]
    physio_labels: List[str]
    physio_basis: np.ndarray
    event_basis: np.ndarray


class PLSEventPhysioModel:
    """
    Partial least squares analysis of task events and physio signals regressed onto functional 
    MRI time courses. 

    Attributes
    ----------
    n_components: int
        Number of PLS components to extract. Defaults to 5.
    physio_lags: int
        The number of lags (in TRs) of the physiological regressor. Defaults to 10.
    regressor_duration: float
        duration of the spline regressors - i.e. the duration after onset of the event. This 
        should be set around the expected duration of the hemodynamic response
        to the event (default: 20.0 seconds).
    n_knots_event: int
        number of knots in the spline basis across temporal lags of the event regressors. 
        Controls the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    n_knots_physio: int
        number of knots in the spline basis across temporal lags of the physio regressor. 
        Controls the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    event_knots: List[int] | None
        knot locations for the spline basis across temporal lags of the event regressors. If supplied, this
        overrides the n_knots_event parameter.
    physio_knots: List[int] | None
        knot locations for the spline basis across temporal lags of the physio regressors. If supplied, this
        overrides the n_knots_physio parameter.
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
        n_components: int = 5,
        physio_lags: int = 10,
        regressor_duration: float = 20.0,
        n_knots_event: int = 5,
        n_knots_physio: int = 5,
        physio_knots: List[int] | None = None,
        event_knots: List[int] | None = None,
        basis: Literal['cr','bs'] = 'bs'
    ):
        self.n_components = n_components
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
    ) -> PLSResults:
        """
        fit PLS model of combined event and physio lag spline basis
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
        self.basis_event = SplineLagBasis(
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
        self.basis_physio = SplineLagBasis(
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

        # z-score all regressors
        event_regs_concat = np.array(zscore(event_regs_concat, axis=0))
        physio_regs_concat = np.array(zscore(physio_regs_concat, axis=0))

        # fit PLS regression model
        self.pls = MBPLS(n_components=self.n_components, max_tol=1e-6, full_svd=False)
        self.pls.fit(
            [event_regs_concat, physio_regs_concat],
            fmri_concat
        ) 

        return PLSResults(
            pls_params=PLSParams(
                n_components=self.n_components,
                physio_lags=self.physio_lags,
                regressor_duration=self.regressor_duration,
                n_knots_event=self.n_knots_event,
                n_knots_physio=self.n_knots_physio,
                basis_type=self.basis_type
            ),
            pls=self.pls,
            physio_col_labels=self.physio_reg_cols,
            event_col_labels=self.event_reg_cols,
            trial_types=self.trial_types,
            physio_labels=self.physio_labels,
            physio_basis=np.array(self.basis_physio.basis),
            event_basis=np.array(self.basis_event.basis)
        )
