"""
Functions for feature extraction from physiological data.
"""

import numpy as np
import neurokit2 as nk


def extract_ppg_features(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract heart rate and PPG amplitude from PPG signal

    Parameters
    ----------
        ts : np.ndarray
            time series of raw PPG signal
        sf : float
            sampling frequency

    Returns
    -------
    dict[str, np.ndarray]
        PPG features
    """
    # extract PPG features (get peaks and rate)
    ppg_df, ppg_info = nk.ppg_process(ts, sampling_rate=sf)
    # PPG Peak Amplitude
    ppg_peaks_loc = np.where(ppg_df['PPG_Peaks'])[0]
    ppg_peaks_amp = np.abs(ppg_df['PPG_Clean'].iloc[ppg_peaks_loc])
    ppg_amp = nk.signal_interpolate(
        ppg_peaks_loc, ppg_peaks_amp,
        np.arange(ppg_df.shape[0]),
        method='cubic'
    )
    return {
        'heart_rate': ppg_df['PPG_Rate'].to_numpy(),
        'ppg_amplitude': np.asarray(ppg_amp),
    }


def extract_resp_features(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract respiratory amplitude and rate by method of Harrison et al. (2021)
    https://doi.org/10.1016/j.neuroimage.2021.117787

    Parameters
    ----------
        ts : np.ndarray
            time series of raw respiratory signal
        sf : float
            sampling frequency

    Returns
    -------
    dict[str, np.ndarray]
        respiratory amplitude and rate signals
    """
    # Clean raw respiratory signal
    resp_features, resp_info = nk.rsp_process(
        ts,
        sampling_rate=sf,
    )
    return {
        'resp_amp': resp_features['RSP_Amplitude'].to_numpy(),
        'resp_rate': resp_features['RSP_Rate'].to_numpy(),
    }

