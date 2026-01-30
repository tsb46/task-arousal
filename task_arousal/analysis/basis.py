"""
Module for basis functions used in modeling.
"""

from typing import List, Literal, Tuple, Dict

import numpy as np
import pandas as pd

from patsy import dmatrix  # type: ignore
from scipy.interpolate import interp1d


class SplineLagBasis:
    """
    Spline basis for modeling temporal lags of a physio signal based on
    scikit-learn fit/transform API. Specifically, a spline basis is fit
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
    tr: float | None
        repetition time of the fMRI data (in seconds). Required for calculating
        knots_per_sec if that parameter is set. Otherwise can be None. (default: None)
    knots_per_sec: float
        number of knots per second in the spline basis across temporal lags. This ensures
        that varying duration trials have similar temporal smoothness in the basis. For example,
        a value of 0.5 results in one knot every 2 seconds. Default is 0.5 knots per second.
    n_knots: int | None
        fix the number of knots in the spline basis across temporal lags. If this parameter is set,
        the knots_per_sec parameter is ignored. Default is None.
    knots: List[int]
        Locations of knots in spline basis across temporal lags. If provided,
        the knots_per_sec and n_knots parameters are ignored.
    basis_type: Literal['ns','bs']
        basis type for the spline basis. 'ns' for natural spline, 'bs' for B-spline.

    """

    def __init__(
        self,
        knots_per_sec: float = 0.5,
        tr: float | None = None,
        n_knots: int | None = None,
        knots: List[int] | None = None,
        basis_type: Literal["cr", "bs"] = "bs",
    ):
        # specify knots parameters
        if (knots is None) & (n_knots is None):
            if tr is None and knots_per_sec is not None:
                raise ValueError("tr must be specified if knots_per_sec is set")
        self.knots_per_sec = knots_per_sec
        self.tr = tr
        self.n_knots = n_knots
        self.knots = knots
        self.basis_type = basis_type

    def create(self, nlags: int, neg_nlags: int = 0) -> "SplineLagBasis":
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
        self.lags = np.arange(neg_nlags, nlags + 1)
        if self.knots is not None:
            self.basis = dmatrix(
                f"{self.basis_type}(x, knots=self.knots) - 1", {"x": self.lags}
            )
        else:
            if self.n_knots is None:
                # calculate number of knots based on knots_per_sec
                if self.tr is None:
                    raise ValueError("tr must be specified if n_knots is None")
                duration_sec = (nlags - neg_nlags) * self.tr
                self._n_knots = max(3, int(np.ceil(duration_sec * self.knots_per_sec)))
            else:
                self._n_knots = self.n_knots

            self.basis = dmatrix(
                f"{self.basis_type}(x, df=self._n_knots) - 1", {"x": self.lags}
            )

        return self

    def project(self, X: np.ndarray, fill_val: float = np.nan) -> np.ndarray:
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
        lag_proj = np.empty((lagmat.shape[0], n_splines), dtype=lagmat.dtype)
        for spline_n in np.arange(n_splines):
            lag_proj[:, spline_n] = np.dot(lagmat, self.basis[:, spline_n])

        return lag_proj


def create_spline_event_reg(
    event_dfs: List[pd.DataFrame],
    outcome_data: List[np.ndarray],
    tr: float,
    resample_tr: float,
    slice_timing_ref: float,
    trial_types: List[str],
    knots_per_sec: float,
    n_knots: int | None,
    basis_type: str,
    knots: List[int] | None,
    regressor_extend: float = 15.0,
    regressor_duration: float | None = None,
) -> Tuple[
    List[np.ndarray],
    Dict[str, int],
    Dict[str, SplineLagBasis],
    Dict[str, float],
    Dict[str, float],
]:
    """
    Utility function to create spline regressors for the task onsets
    of each trial across all sessions/runs.

    Parameters
    ----------
    event_dfs: List[pd.DataFrame]
        List of dataframes containing event information (onset, duration, etc.).
    outcome_data: List[np.ndarray]
        List of fMRI 2D datasets (2D - time x voxels) or physio signals (2D - time x signals).
        This should be in the same order as event_dfs (i.e., outcome_data[i] corresponds to event_dfs[i]).
        This should not be concatenated data across runs/sessions.
    tr: float
        repetition time of the fMRI data (in seconds)
    resample_tr: float
        time resolution for resampling the event time course (in seconds)
    slice_timing_ref: float
        slice timing reference (in seconds)
    trial_types: List[str]
        List of unique trial types across all event_dfs.
    knots_per_sec: float
        number of knots per second in the spline basis across temporal lags. This ensures
        that varying duration trials have similar temporal smoothness in the basis. For example,
        a value of 0.5 results in one knot every 2 seconds. Default is 0.5 knots per second.
    n_knots: int | None
        fix the number of knots in the spline basis across temporal lags. If this parameter is set,
        the knots_per_sec parameter is ignored. Default is None.
    basis_type: Literal['cr','bs']
        basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.
    knots: List[int] | None
        knot locations for the spline basis across temporal lags. If supplied, this
        overrides the n_knots parameter. If this parameter is set, the knots_per_sec parameter and
        n_knots parameter are ignored.
    regressor_extend: float
        how much time (in seconds) after the end of the event to extend the regressor. If None, the regressor
        will only cover the duration of the event. Defaults is 10 seconds. If regressor_duration is set, this parameter is ignored.
    regressor_duration: float | None
        fix the duration of all spline regressors - i.e. the duration after onset of the event.
        If set to None, the regressor duration will be set to the event duration from the event data.
        Note, that if regressor_duration is None, the number of lags (nlags) will vary across events.
    """
    # initialize basis metadata to be filled with trial-specific values
    nlags = {}
    basis = {}
    # loop through event dataframes to get the max duration for each trial type
    # across all sessions/runs
    trial_durations_dict = {}
    trial_durations_extend_dict = {}

    for trial in trial_types:
        max_duration = 0.0
        for event_df in event_dfs:
            trial_durations = event_df[event_df["trial_type"] == trial][
                "duration"
            ].to_numpy()
            if len(trial_durations) > 0:
                trial_max = np.max(trial_durations)
                # update max duration if trial_max is greater
                if trial_max > max_duration:
                    max_duration = trial_max
        trial_durations_dict[trial] = max_duration
        # extend regressor duration by regressor_extend
        if regressor_duration is None:
            trial_durations_extend_dict[trial] = max_duration + regressor_extend
        else:
            trial_durations_extend_dict[trial] = regressor_duration
        # calculate number of lags based on regressor duration and TR
        nlags[trial] = int(np.ceil(trial_durations_extend_dict[trial] / resample_tr))
        # create spline basis
        basis[trial] = SplineLagBasis(
            knots_per_sec=knots_per_sec,
            tr=resample_tr,
            n_knots=n_knots,
            knots=knots,
            basis_type=basis_type,  # type: ignore
        )

        basis[trial].create(nlags[trial], neg_nlags=0)

    # create event regressors for each session/run
    event_regs = []
    for i, (event_df, outcome_d) in enumerate(zip(event_dfs, outcome_data)):
        n_vols = outcome_d.shape[0]
        # create boxcar event regressor resampled at RESAMPLE_TR
        event_reg, frametimes, h_frametimes = boxcar(
            event_df,
            tr=tr,
            resample_tr=resample_tr,
            n_vols=n_vols,
            slicetime_ref=slice_timing_ref,
            trial_types=trial_types,
            impulse_dur=0.5,
        )
        # project each trial event regressor onto spline basis
        events_regs_trial = []
        for t, trial in enumerate(trial_types):
            # project event regressor onto spline basis
            # fill in NaNs with 0 to keep same length
            event_reg_proj = basis[trial].project(event_reg[t], fill_val=0.0)
            # downsample (interpolate) event regressor to match fmri times
            interp_func = interp1d(h_frametimes, event_reg_proj.T, kind="cubic")
            event_reg_proj = interp_func(frametimes).T
            # trim fmri_img and event_reg to same length
            events_regs_trial.append(event_reg_proj)
        # create design matrix by concatenating trial event regressors
        event_regs_trial = np.hstack(events_regs_trial)
        event_regs.append(event_regs_trial)

    return event_regs, nlags, basis, trial_durations_dict, trial_durations_extend_dict


def boxcar(
    event_df: pd.DataFrame,
    tr: float,
    resample_tr: float,
    n_vols: int,
    slicetime_ref: float,
    trial_types: List[str],
    impulse_dur: float = 0.1,
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
    frametimes = np.linspace(slicetime_ref, (n_vols - 1 + slicetime_ref) * tr, n_vols)

    # Create index based on resampled tr
    h_frametimes = np.arange(0, frametimes[-1] + 1, resample_tr)

    # loop through trial_types and create boxcar function
    trial_event_ts = []
    for trial_type in trial_types:
        df_trial = event_df[event_df["trial_type"] == trial_type].copy()
        # initialize zero vector for event time course
        event_ts = np.zeros_like(h_frametimes).astype(np.float64)

        # Grab onsets from event_df
        onsets = df_trial["onset"].to_numpy()
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
            np.searchsorted(h_frametimes, onsets + impulse_dur), tmax - 1
        )
        # fill in boxcar by setting samples between onset and offset to 1
        for t in zip(t_offset):
            event_ts[t] -= 1

        # cumulative sum to create boxcar function
        event_ts = np.cumsum(event_ts)
        trial_event_ts.append(event_ts.reshape(-1, 1))

    return trial_event_ts, frametimes, h_frametimes


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
    x_lag = np.empty(shape=(n_rows, n_cols * n_lags), order="F", dtype=x.dtype)
    # fill w/ Nans
    x_lag[:] = fill_val
    # Copy lagged columns of X into X_lag
    for i, lag in enumerate(lags):
        # target columns of X_lag
        j = i * n_cols
        k = j + n_cols  # (i+1) * ncols
        # number rows of X
        nl = n_rows - abs(lag)
        # Copy
        if lag >= 0:
            x_lag[lag:, j:k] = x[:nl, :]
        else:
            x_lag[:lag, j:k] = x[-nl:, :]
    return x_lag
