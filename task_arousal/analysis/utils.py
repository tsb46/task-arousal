"""
Utility functions for analysis module
"""
from typing import List, Tuple

import numpy as np
import pandas as pd


def boxcar(
    event_df: pd.DataFrame, 
    tr: float, 
    resample_tr: float, 
    n_vols: int, 
    slicetime_ref: float, 
    trial_types: List[str],
    impulse_dur: float = 0.1
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Create a boxcar (rectangular) function time series.

    Parameters:
    ----------
        event_df: pd.DataFrame
            DataFrame with 'onset' and 'duration' columns.
        tr: float
            Repetition time of the fMRI scan (in seconds).
        resample_tr: float
            Time resolution for resampling the event time course (in seconds).
        n_vols: int
            Number of volumes in the fMRI scan.
        slicetime_ref: float
            Slice timing reference (in seconds).
        trial_types: List[str]
            List of unique trial types.
        impulse_dur: float, optional
            Duration of the boxcar impulse (in seconds). Defaults to 0.1.

    Returns:
    -------
        trial_event_ts: List[np.ndarray]
            List of arrays, each of shape (n_timepoints, 1) with boxcar functions for each trial type.
        frametimes: np.ndarray
            Time points of the original fMRI scan (in seconds).
        h_frametimes: np.ndarray
            Time points of the resampled fMRI scan (in seconds).
    """
    # get time samples of functional scan based on slicetime reference
    frametimes = np.linspace(
        slicetime_ref, 
        (n_vols - 1 + slicetime_ref) * tr, 
        n_vols
    )

    # Create index based on resampled tr
    h_frametimes = np.arange(0, frametimes[-1]+1, resample_tr)

    # loop through trial_types and create boxcar function
    trial_event_ts = []
    for trial_type in trial_types:
        df_trial = event_df[event_df['trial_type'] == trial_type].copy()
        # initialize zero vector for event time course
        event_ts = np.zeros_like(h_frametimes).astype(np.float64)

        # Grab onsets from event_df
        onsets = df_trial['onset'].to_numpy()
        # create unit impulses at event onsets
        # initialize zero vector for event time course
        event_ts = np.zeros_like(h_frametimes).astype(np.float64)
        # maximum index for event time course
        tmax = len(h_frametimes)
        # Get samples nearest to onsets
        t_onset = np.minimum(np.searchsorted(h_frametimes, onsets), tmax - 1)
        for t in t_onset:
            event_ts[t] = 1
        # get samples nearest to offsets
        t_offset = np.minimum(
            np.searchsorted(h_frametimes, onsets + impulse_dur), 
            tmax - 1
        )
        # fill in boxcar by setting samples between onset and offset to 1
        for t in zip(t_offset):
            event_ts[t] -= 1
    
        # cumulative sum to create boxcar function
        event_ts = np.cumsum(event_ts)
        trial_event_ts.append(event_ts.reshape(-1,1))

    return trial_event_ts, frametimes, h_frametimes


def create_interaction_matrix(
    event_regs: np.ndarray,
    physio_reg: np.ndarray
) -> np.ndarray:
    """
    create interaction matrix between event regressors and physio regressor

    Parameters
    ----------
    event_regs: np.ndarray
        Event regressors (2D - time x event regressors)
    physio_reg: np.ndarray
        Physio regressor (2D - time x physio regressors)

    Returns
    -------
    interaction_mat: np.ndarray
        Interaction matrix between event and physio regressors
        (2D - time x (event regressors * physio regressors))
    """
    n_time = event_regs.shape[0]
    n_event_regs = event_regs.shape[1]
    n_physio_regs = physio_reg.shape[1]
    # allocate memory for interaction matrix
    interaction_mat = np.empty(
        (n_time, n_event_regs * n_physio_regs),
        dtype=event_regs.dtype
    )
    for i in range(n_event_regs):
        for j in range(n_physio_regs):
            interaction_mat[:, i*n_physio_regs + j] = (
                event_regs[:, i] * physio_reg[:, j]
            )
    
    return interaction_mat


def lag_mat(x: np.ndarray, lags: list[int], fill_val: float = np.nan) -> np.ndarray:
    """
    Create array of time-lagged copies of the time course. Modified
    for negative lags from:
    https://github.com/ulf1/lagmat

    Parameters
    ----------
        x : np.ndarray
            The time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        lags : list[int]
            List of integer lags (shifts) to apply to the time course.
            Positive values indicate a lag (shift down), negative values
            indicate a lead (shift up).
        fill_val : float, optional
            Value to use for filling in missing values after shifting.
            Defaults to np.nan.

    """
    n_rows, n_cols = x.shape
    n_lags = len(lags)
    # allocate memory
    x_lag = np.empty(
        shape=(n_rows, n_cols * n_lags),
        order='F', dtype=x.dtype
    )
    # fill w/ Nans
    x_lag[:] = fill_val
    # Copy lagged columns of X into X_lag
    for i, l in enumerate(lags):
        # target columns of X_lag
        j = i * n_cols
        k = j + n_cols  # (i+1) * ncols
        # number rows of X
        nl = n_rows - abs(l)
        # Copy
        if l >= 0:
            x_lag[l:, j:k] = x[:nl, :]
        else:
            x_lag[:l, j:k] = x[-nl:, :]
    return x_lag
