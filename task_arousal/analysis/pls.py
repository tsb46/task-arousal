"""
Partial Least Squares regression analysis module.
"""
from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import zscore

from task_arousal.constants import TR, SLICE_TIMING_REF, EVENT_COLUMNS
from task_arousal.analysis.dlm import BSplineLagBasis
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
    n_physio_knots: int
    basis_type: str

@dataclass
class PLSResults:
    pls_params: PLSParams
    pls: PLSRegression
    reg_col_labels: List[str]
    trial_types: List[str]

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
        physio_data: List[np.ndarray]
    ) -> PLSResults:
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
        physio_data: List[np.ndarray]
            List of physio signal (2D - time x 1).
            This should be in the same order as event_dfs (i.e., physio_data[i] corresponds to event_dfs[i]).
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
        if len(event_dfs) != len(physio_data):
            raise ValueError("event_dfs and physio_data must have the same length")
        # check that physio_data have required shape
        for i, physio in enumerate(physio_data):
            if physio.ndim != 2 or physio.shape[1] != 1:
                raise ValueError(f"Invalid shape for physio_data {i}: {physio.shape}")
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
        event_reg_cols = [
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
        # define physio regressor column names
        physio_reg_cols = [
            f"physio_lag_spline{n+1}"
            for n in range(self.basis_physio.basis.shape[1])
        ]
        # create physio regressor for each session/run
        self.physio_regs = []
        for i, physio_d in enumerate(physio_data):
            if physio_d.shape[0] != fmri_data[i].shape[0]:
                raise ValueError(f"physio_data {i} and fmri_data {i} must have the same number of time points")
            # project physio signal lags on B-spline basis
            physio_reg_proj = self.basis_physio.project(physio_d, fill_val=0.0)
            self.physio_regs.append(physio_reg_proj)

        # concatenate fmri data across sessions/runs
        fmri_concat = np.vstack(fmri_data)
        # concatenate event regressors across sessions/runs
        event_regs_concat = np.vstack(self.event_regs)
        # concatenate physio regressors across sessions/runs
        physio_regs_concat = np.vstack(self.physio_regs)

        # concatenate all regressors
        all_regs_concat = np.hstack((
            event_regs_concat,
            physio_regs_concat,
        ))
        # define all regressor column names
        self.reg_col_labels = event_reg_cols + physio_reg_cols

        # z-score all regressors
        all_regs_concat = np.array(zscore(all_regs_concat, axis=0))

        # fit PLS regression model
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(
            all_regs_concat,
            fmri_concat
        )

        return PLSResults(
            pls_params=PLSParams(
                n_components=self.n_components,
                physio_lags=self.physio_lags,
                regressor_duration=self.regressor_duration,
                n_knots_event=self.n_knots_event,
                n_physio_knots=self.n_knots_physio,
                basis_type=self.basis_type
            ),
            pls=self.pls,
            reg_col_labels=self.reg_col_labels,
            trial_types=self.trial_types
        )

        # # Form pairwise latent interactions
        # T = self.pls.x_scores_
        # latent_interactions = []
        # names_inter = []
        # for i in range(T.shape[1]):
        #     for j in range(i, T.shape[1]):
        #         latent_interactions.append(T[:, i] * T[:, j])
        #         names_inter.append(f"T{i}_x_T{j}")
        # Z = np.column_stack([T] + latent_interactions)

        # # scale Z
        # Z = np.array(zscore(Z, axis=0))
        # # Second-stage PLS or linear model predicting y from Z
        # pls2 = PLSRegression(n_components=min(5, Z.shape[1])) 
        # pls2.fit(Z, fmri_concat)
        # return self
