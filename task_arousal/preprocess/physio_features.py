"""
Functions for feature extraction from physiological data.
"""

from sys import platlibdir
from typing import Literal

import numpy as np
import neurokit2 as nk

from scipy.signal import find_peaks


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
    # find peaks of PPG signal
    ppg_peaks_loc = np.where(ppg_df['PPG_Peaks'])[0]
    # get peak amplitudes and interpolate
    ppg_peaks_amp = np.abs(ppg_df['PPG_Clean'].iloc[ppg_peaks_loc])
    ppg_amp = nk.signal_interpolate(
        ppg_peaks_loc, ppg_peaks_amp,
        np.arange(ppg_df.shape[0]),
        method='monotone_cubic'
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
        'resp_rvt': resp_features['RSP_RVT'].to_numpy()
    }


def extract_resp_co2_features(
    ts: np.ndarray, 
    sf: int,
) -> dict[str, np.ndarray]:
    """
    Extract end-tidal CO2 waveforms from raw
    CO2 recordings through peak detection and interpolation

    Parameters
    ----------
        ts : np.ndarray
            time series of raw respiratory CO2 signal
        sf : float
            sampling frequency

    Returns
    -------
    dict[str, np.ndarray]
        respiratory end-tidal CO2 signal
    """
    # band-pass filter the CO2 signal to match typical breathing frequencies
    ts_filt = nk.signal_filter(
        ts,
        sampling_rate=sf,
        lowcut=0.1,
        highcut=0.4,
        method='butterworth',
        order=4,
    )
    # find peaks (end-tidal CO2 points)
    co2_peaks, peaks_info = find_peaks(
        ts_filt,
        height=np.percentile(ts_filt, 50),  # only consider peaks above 50th percentile
        distance=sf*1.5,  # minimum distance of 1.5 seconds between peaks
    )
    endtidal_co2 = nk.signal_interpolate(
        co2_peaks, peaks_info['peak_heights'],
        np.arange(len(ts_filt)),
        method='monotone_cubic'
    )
    return {
        'endtidal_co2': endtidal_co2
    }


def extract_resp_o2_features(
    ts: np.ndarray, 
    sf: int,
) -> dict[str, np.ndarray]:
    """
    Extract end-tidal O2 waveforms from raw
    O2 recordings through peak detection and interpolation

    Parameters
    ----------
        ts : np.ndarray
            time series of raw respiratory O2 signal
        sf : float
            sampling frequency

    Returns
    -------
    dict[str, np.ndarray]
        respiratory end-tidal O2 signal
    """
    # band-pass filter the O2 signal to match typical breathing frequencies
    ts_filt = nk.signal_filter(
        ts,
        sampling_rate=sf,
        lowcut=0.1,
        highcut=0.4,
        method='butterworth',
        order=4,
    )
    # find peaks (end-tidal O2 points)
    o2_peaks, peaks_info = find_peaks(
        ts_filt,
        height=np.percentile(ts_filt, 50),  # only consider peaks above 50th percentile
        distance=sf*1.5,  # minimum distance of 1.5 seconds between peaks
    )
    # the O2 signal has a prominent transient artifact at the first peak of the recording
    # so we will set the first peak equal to the second peak
    if len(o2_peaks) > 1:
        peaks_info['peak_heights'][0] = peaks_info['peak_heights'][1]

    endtidal_o2 = nk.signal_interpolate(
        o2_peaks, peaks_info['peak_heights'],
        np.arange(len(ts_filt)),
        method='monotone_cubic'
    )

    return {
        'endtidal_o2': endtidal_o2
    }

