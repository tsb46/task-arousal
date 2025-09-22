"""
First-level GLM analysis using Nilearn.
"""
from dataclasses import dataclass
from typing import Literal, List

import pandas as pd
import nibabel as nib
import numpy as np

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs
from sklearn.linear_model import Ridge

from task_arousal.analysis.utils import create_interaction_matrix, lag_mat
from task_arousal.constants import TR, SLICE_TIMING_REF, MASK, EVENT_COLUMNS


@dataclass
class GLMResults:
    """
    Class for storing results of GLM analysis. 

    Attributes
    ----------
    design_matrix: pd.DataFrame
        the design matrix used in the GLM
    contrast_maps: dict[str, nib.Nifti1Image]
        a dictionary of contrast maps for each condition in the design matrix
    hrf: str
        the hemodynamic response function used in the GLM

    """
    contrast_maps: dict[str, nib.Nifti1Image] # type: ignore
    hrf: str
    fir_delays: np.ndarray | None = None


@dataclass
class GLMPhysioResults:
    """
    Class for storing results of GLM analysis. 

    Attributes
    ----------
    design_matrix: pd.DataFrame
        the design matrix used in the GLM
    contrast_maps: dict[str, nib.Nifti1Image]
        a dictionary of contrast maps for each condition in the design matrix
    hrf: str
        the hemodynamic response function used in the GLM
    """
    pred_func: np.ndarray
    physio_val: List[float]
    physio_lag: int
    design_matrix_cols: List[str]
    hrf: str

class GLM:
    """
    Fixed-effects general linear model (GLM) implementation using Nilearn.
    """
    def __init__(
        self, 
        hrf: Literal['spm', 'glover', 'fir'] = 'spm',
        fir_delays: np.ndarray | None = None
    ):
        """Initialize GLM.

        Attributes
        ----------
           hrf : Literal['spm', 'glover', 'fir']
                The hemodynamic response function to use. Defaults to 'spm'.
            fir_delays : np.ndarray | None
                If using 'fir' hrf, the delays (in seconds) to use for the
                finite impulse response model. Must be specified if hrf is 'fir'.
        """
        self.hrf = hrf
        self.fir_delays = fir_delays
        if self.hrf == 'fir' and self.fir_delays is None:
            raise ValueError("fir_delays must be specified if hrf is 'fir'")

    def fit(self, event_df: List[pd.DataFrame], fmri: List[nib.Nifti1Image]) -> GLMResults: # type: ignore
        """
        Fit the GLM model to the event-related design matrix.

        Parameters
        ----------
            event_df : List[pd.DataFrame]
                A list of event timing dataframes with 'onset', 'duration', 
                and 'trial_type' columns (corresponding to multiple sessions/runs).
            fmri : List[nib.Nifti1Image]
                A list of fMRI scans (corresponding to multiple sessions/runs). These
                should be in the same order as event_df (i.e., fmri[i] corresponds to event_df[i]).

        Returns
        -------
            GLMResults
                The results of the GLM analysis.
        """
        # check that event_df has required columns
        for i, df in enumerate(event_df):
            if not all(col in df.columns for col in EVENT_COLUMNS):
                raise ValueError(f"Missing columns: {EVENT_COLUMNS} in dataframe {i}")

        # assumes all event dfs have same trial types
        trial_labels = event_df[0]['trial_type'].unique().tolist()
        # create design matrices for each session
        design_matrices = []
        for df, fmri_img in zip(event_df, fmri):
            # get number of time points and TR from fmri image
            fmri_n_tp = fmri_img.shape[-1]
            # create frame times based on the number of scans and TR
            fmri_times = np.arange(0, fmri_n_tp*TR, TR)
            fmri_times += (TR * SLICE_TIMING_REF)  # middle of the TR
            # create design matrix
            design_matrix = make_first_level_design_matrix(
                fmri_times,
                df,
                hrf_model=self.hrf,
                drift_model=None, # type: ignore
                fir_delays=self.fir_delays
            )
            design_matrices.append(design_matrix)

        # load mask
        mask_img = nib.load(MASK) # type: ignore

        # fit the GLM model
        self.model = FirstLevelModel(
            hrf_model=self.hrf,
            mask_img=mask_img,
            signal_scaling=False
        )
        model = self.model.fit(fmri, design_matrices=design_matrices)

        if self.hrf == 'fir' and self.fir_delays is not None:
            # create contrast maps for each trial type by combining FIR delays
            contrast_maps = {
                label: concat_imgs([ # type: ignore
                    model.compute_contrast(f"{label}_delay_{delay}", output_type='z_score')
                    for delay in self.fir_delays
                ], ensure_ndim=4)
                for label in trial_labels
            }
        else:
            contrast_maps = {
                label: model.compute_contrast(label, output_type='z_score')
                for label in trial_labels
            }

        return GLMResults(
            contrast_maps=contrast_maps, # type: ignore
            hrf=self.hrf,
            fir_delays=self.fir_delays
        )
    

class GLMPhysio:
    """
    Custom GLM that includes physiological regressors, and their interactions with task events.
    Only 'spm' and 'glover' HRFs are currently supported.
    """
    def __init__(
        self,
        physio_lag: int = 5,
        hrf: Literal['spm', 'glover'] = 'spm',
    ):
        """Initialize GLM.

        Attributes
        ----------
            physio_lag : int
                The chosen lag (in TRs) to include for the physiological regressor. Defaults to 5.
            hrf : Literal['spm', 'glover']
                The hemodynamic response function to use. Defaults to 'spm'.
        """
        self.physio_lag = physio_lag
        self.hrf = hrf

    def fit(
        self, 
        event_df: List[pd.DataFrame], 
        fmri: List[np.ndarray],
        physio: List[np.ndarray]
    ) -> None:
        """
        Fit the GLM model to the event-related design matrix.

        Parameters
        ----------
            event_df : List[pd.DataFrame]
                A list of event timing dataframes with 'onset', 'duration', 
                and 'trial_type' columns (corresponding to multiple sessions/runs).
            fmri : List[np.ndarray]
                A list of fMRI datasets (2D - time x voxels) corresponding to multiple sessions/runs. These
                should be in the same order as event_df (i.e., fmri[i] corresponds to event_df[i]).
            physio: List[np.ndarray]
                List of physio signal (2D - time x 1).
                This should be in the same order as event_dfs (i.e., physio[i] corresponds to event_df[i]).

        Returns
        -------
            GLMResults
                The results of the GLM analysis.
        """
        # check that event_df has required columns
        for i, df in enumerate(event_df):
            if not all(col in df.columns for col in EVENT_COLUMNS):
                raise ValueError(f"Missing columns: {EVENT_COLUMNS} in dataframe {i}")

        # create design matrices for each session
        design_matrices = []
        for df, fmri_d in zip(event_df, fmri):
            # get number of time points and TR from fmri dataset
            fmri_n_tp = fmri_d.shape[0]
            # create frame times based on the number of scans and TR
            fmri_times = np.arange(0, fmri_n_tp*TR, TR)
            fmri_times += (TR * SLICE_TIMING_REF)  # middle of the TR
 
            # create design matrix
            design_matrix = make_first_level_design_matrix(
                fmri_times,
                df,
                hrf_model=self.hrf,
                drift_model=None # type: ignore
            )
            # apply z-score normalization to design matrix
            # get indices of non-constant columns
            non_const_cols = design_matrix.columns[design_matrix.nunique() > 1]
            design_matrix[non_const_cols] = (
                design_matrix[non_const_cols] - design_matrix[non_const_cols].mean(axis=0)
            )
            design_matrix[non_const_cols] = (
                design_matrix[non_const_cols] / design_matrix[non_const_cols].std(axis=0)
            )
            design_matrices.append(design_matrix)

        # save column names of design matrix
        self.design_matrix_cols = design_matrices[0].columns.tolist()
        # concatenate fmri data across sessions/runs
        fmri_concat = np.vstack(fmri)
        # concatenate event regressors across sessions/runs
        event_regs_concat = np.vstack([d.to_numpy() for d in design_matrices])
        # create lagged physio regressors for each session/run
        physio_lags = [lag_mat(physio_d, [self.physio_lag], fill_val=0) for physio_d in physio]
        physio_lags_concat = np.vstack(physio_lags)

        # create interaction matrix between event and physio regressors
        interaction_regs_concat = create_interaction_matrix(
            event_regs_concat,
            physio_lags_concat
        )
        # combine event, physio, and interaction regressors into single design matrix
        design_matrix_concat = np.hstack([
            event_regs_concat,
            physio_lags_concat,
            interaction_regs_concat
        ])
        
        self.model = Ridge(alpha=1.0)
        self.model.fit(design_matrix_concat, fmri_concat) # type: ignore

    def evaluate(
        self, 
        trial: str,
        physio_val_min = -2.0,
        physio_val_max = 2.0,
        physio_val_steps = 10
    ) -> GLMPhysioResults:
        """
        Evaluate the fitted GLM model to get predicted fMRI response.

        Parameters
        ----------
            trial : str
                The trial type to evaluate. Must be one of the trial types in the design matrix.
            physio_val_min : float
                The minimum value to use for the physio regressor. Defaults to -3.0.
            physio_val_max : float
                The maximum value to use for the physio regressor. Defaults to 3.0.
            physio_val_steps : int
                The number of steps to use for the physio regressor. Defaults to 50.

        Returns
        -------
            GLMPhysioResults
                The results of the GLM analysis.
        """
        PRED_EVENT_VAL = 4.0  # fixed event regressor value for prediction
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been fitted yet. Call fit() before evaluate().")

        # get index of trial in design matrix columns
        if trial not in self.design_matrix_cols:
            raise ValueError(f"Trial {trial} not found in design matrix columns.")
        trial_idx = self.design_matrix_cols.index(trial)
        # get number of event and physio regressors
        n_event_regs = len(self.design_matrix_cols)
        
        if trial not in self.design_matrix_cols:
            raise ValueError(f"Trial {trial} not found in design matrix columns.")
        # create array of physio values to evaluate
        physio_val = np.linspace(physio_val_min, physio_val_max, physio_val_steps)
        # find index of trial in design matrix columns
        trial_idx = self.design_matrix_cols.index(trial)
        # create event regressor for single trial type
        event_pred = np.zeros((len(physio_val), n_event_regs))
        event_pred[:, trial_idx] = PRED_EVENT_VAL

        # create physio regressor
        physio_pred = np.zeros((len(physio_val), 1))
        for i, val in enumerate(physio_val):
            physio_pred[i, 0] = val

        # create interaction regressor
        interaction_pred = create_interaction_matrix(
            event_pred,
            physio_pred
        )
        # create full design matrix for single trial type
        full_pred = np.hstack([
            event_pred,
            physio_pred,
            interaction_pred
        ])

        pred_func = self.model.predict(full_pred)

        return GLMPhysioResults(
            pred_func=pred_func,
            physio_val=physio_val.tolist(),
            physio_lag=self.physio_lag,
            design_matrix_cols=self.design_matrix_cols,
            hrf=self.hrf
        )
    