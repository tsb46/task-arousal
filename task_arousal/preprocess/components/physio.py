"""
Physiological signal processing components
"""

import json

from typing import Tuple, List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import neurokit2 as nk
import pandas as pd


from scipy.signal import find_peaks
from scipy.stats import zscore


def physio_pipeline(
    dataset: str,
    physio_fp: str | Tuple[str, str],
    physio_json: str | None,
    fmri_fp: str,
    physio_cols: List[str],
    tr: float,
    physio_resample_f: float,
    fmri_dummy_n: int,
    highpass: float,
    remove_dummy: bool = True,
    slicetiming_ref: float = 0.5,
    save_physio_figs: bool = False,
) -> dict[str, np.ndarray]:
    """
    Physiological pipeline for processing physiological data.

    Preprocessing steps:

    1) Resample to 50Hz (polyphase filtering to avoid aliasing)
    2) Feature extraction
    3) Resample to fMRI time points (low-pass filter to avoid aliasing, then interpolate)
    4) High-pass filtering to match fMRI highpass (> 0.01 Hz)
    5) Standardization

    Parameters
    ----------
    dataset : str
        The dataset identifier.
    physio_fp : str | Tuple[str, str]
        The file path to the physiological data. For datasets with separate files for different physiological signals,
        this can be a tuple of file paths.
    physio_json : str | None
        The file path to the physiological JSON sidecar for euskalibur dataset. HCP physio
        does not have JSON sidecar.
    fmri_fp : str
        The file path to the functional MRI data (to get number of time points for resampling).
    physio_cols : List[str]
        The list of physiological columns to process.
    tr : float
        The repetition time (TR) of the fMRI data.
    physio_resample_f : float
        The target resampling frequency for physiological data (e.g., 50Hz).
    fmri_dummy_n : int
        The number of dummy volumes removed from the fMRI data.
    highpass : float
        The high-pass filter cutoff frequency in Hz.
    remove_dummy : bool, optional
        Whether to remove dummy volumes from the physiological data. Default is True.
    slicetiming_ref : float, optional
        The slice timing reference (0-1) within the TR for resampling physiological data
        to fMRI time points. Default is 0.5 (middle of the TR). Note, NSD data is already resampled
        to the beginning of the TR, so for NSD this will be set to 0.0 in the pipeline.
    save_physio_figs : bool, optional
        Whether to save figures of the physiological signals, by default False.

    Returns
    -------
    dict[str, np.ndarray]
        The processed physiological data.
    """
    # fix slice timing reference for NSD, which is already resampled to the beginning of the TR
    if dataset == "nsd":
        slicetiming_ref = 0.0

    # Load physiological data
    physio_dict, physio_sf = load_physio(
        dataset=dataset,
        physio_fp=physio_fp,
        physio_json=physio_json,
        dummy_n=fmri_dummy_n,
        tr=tr,
        physio_cols=physio_cols,
        remove_dummy=remove_dummy,
    )
    # load fmri data
    fmri_img = nib.load(fmri_fp)  # type: ignore

    # calculate number of time points in fMRI data after dummy volume removal, if applicable
    if remove_dummy:
        fmri_n_tp = fmri_img.shape[-1] - fmri_dummy_n  # type: ignore
    else:
        fmri_n_tp = fmri_img.shape[-1]  # type: ignore

    # define function for feature extraction based on column name
    feature_extraction_funcs = {
        "respiratory_effort": extract_resp_features,
        "cardiac": extract_ppg_features,
    }
    # collections of CO2 and O2 signals are unique to EuskalIBUR dataset, so we will only add these feature extraction functions for this dataset
    if dataset == "euskalibur":
        if "co2" in physio_cols:
            feature_extraction_funcs["co2"] = extract_resp_co2_features
        if "o2" in physio_cols:
            feature_extraction_funcs["o2"] = extract_resp_o2_features

    # loop through physio columns and extract features
    physio_data_proc = {}
    for col in physio_cols:
        # if physio has high sampling rate (1000Hz or 400Hz), so we can downsample to 50Hz
        # use polyphase filtering to avoid aliasing
        physio_resampled = nk.signal_resample(
            physio_dict[col],
            sampling_rate=physio_sf,
            desired_sampling_rate=physio_resample_f,
            method="poly",
        )
        if save_physio_figs:
            # save figure of resampled physio signal
            if dataset == "euskalibur":
                assert isinstance(physio_fp, str), (
                    "physio_fp should be a string for euskalibur dataset"
                )
                fig_fp = physio_fp.replace(".tsv.gz", f"_{col}.png")
            elif dataset == "nsd":
                assert isinstance(physio_fp, tuple), (
                    "physio_fp should be a tuple for nsd dataset"
                )
                fig_fp = physio_fp[0].replace(".tsv", f"_{col}.png")
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            _physio_write_image(
                fp_out=fig_fp,
                ts=physio_resampled,  # type: ignore
                sf=physio_resample_f,
                label=col,
            )

        # extract features
        physio_resampled = np.asarray(physio_resampled)  # ensure numpy array

        if col in feature_extraction_funcs:
            physio_features = feature_extraction_funcs[col](
                physio_resampled, physio_resample_f
            )
            # loop through features and resample to fMRI time points
            for feat_name, feat_ts in physio_features.items():
                # first, low-pass filter the time series to avoid aliasing
                feat_ts_lowpass = nk.signal_filter(
                    feat_ts,
                    sampling_rate=physio_resample_f,  # type: ignore
                    highcut=(1 / (2 * tr)) + 0.05,  # slightly above nyquist frequency
                    method="butterworth",
                    order=4,
                )
                # next, interpolate to fMRI time points
                feat_ts_resampled = _physio_resample_to_fmri(
                    feat_ts_lowpass,  # type: ignore
                    physio_sf=physio_resample_f,
                    fmri_n_tp=fmri_n_tp,
                    fmri_tr=tr,
                    slicetiming_ref=slicetiming_ref,
                )
                # finally, high-pass filter to match fMRI highpass
                feat_ts_bandpassed = nk.signal_filter(
                    feat_ts_resampled,
                    sampling_rate=1 / tr,  # type: ignore
                    lowcut=highpass,
                    method="butterworth",
                    order=4,
                )
                # standardize the signal (z-score)
                feat_ts_bandpassed = zscore(feat_ts_bandpassed)
                # add to processed physio dict
                physio_data_proc[f"{feat_name}"] = feat_ts_bandpassed
        else:
            print(
                f"No feature extraction function defined for column '{col}'. Skipping."
            )

    return physio_data_proc


def load_physio(
    dataset: str,
    physio_fp: str | Tuple[str, str],
    physio_json: str | None,
    dummy_n: int,
    tr: float,
    physio_cols: List[str],
    remove_dummy: bool = True,
) -> Tuple[dict[str, list[float]], float]:
    """
    Load physiological data from a file (HCP or Euskalibur datasets), and trim to start and
    end from the first trigger.

    Parameters
    ----------
    dataset : str
        The dataset identifier.
    physio_fp : str | Tuple[str, str]
        The file path to the physiological data. For datasets with separate files for different physiological signals, this can be a tuple of file paths.
    physio_json : str | None
        The file path to the physiological JSON sidecar for the euskalibur dataset. HCP physio
        does not have JSON sidecar.
    dummy_n : int
        The number of dummy volumes removed from the fMRI data.
    tr : float
        The repetition time (TR) of the fMRI data.
    physio_cols : List[str]
        The list of physiological columns to load.
    remove_dummy : bool, optional
        Whether to remove dummy volumes from the physiological data. Default is True.

    Returns
    -------
    dict[str, float]
        A dictionary containing each physiological label as a key
        and its corresponding time series as a list of floats.
    float
        the sampling frequency of the physiological data.
    """
    # load physio data
    if dataset == "euskalibur":
        if physio_json is None:
            raise ValueError(
                "Euskalibur dataset requires a JSON sidecar for physiological data."
            )
        if isinstance(physio_fp, tuple):
            raise ValueError(
                "Euskalibur dataset should have a single physiological data file, not separate files for different signals."
            )
        physio_df, sampling_freq = _load_physio_euskalibur(
            physio_fp, physio_json, physio_cols
        )
    elif dataset == "nsd":
        if isinstance(physio_fp, str):
            raise ValueError(
                "NSD dataset should have separate physiological data files for pulse and respiration."
            )
        physio_df, sampling_freq = _load_physio_nsd(physio_fp)
        # check that physio_cols are in the dataframe
        missing_cols = [col for col in physio_cols if col not in physio_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected columns in physiological data: {missing_cols}"
            )
        # select only the specified columns
        physio_df = physio_df[physio_cols].copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # trim physio signals to match removal of fMRI dummy volumes
    if remove_dummy:
        physio_df = _trim_physio_signals_dummy(physio_df, dummy_n, tr, sampling_freq)
    # convert to dict
    try:
        physio_dict = physio_df.to_dict(orient="list")
    except KeyError as e:
        raise ValueError(f"Missing expected column in physiological data: {e}")
    return physio_dict, sampling_freq  # type: ignore


def extract_ppg_features(ts: np.ndarray, sf: float) -> dict[str, np.ndarray]:
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
    ppg_df, ppg_info = nk.ppg_process(ts, sampling_rate=sf)  # type: ignore
    # PPG Peak Amplitude
    # find peaks of PPG signal
    ppg_peaks_loc = np.where(ppg_df["PPG_Peaks"])[0]
    # get peak amplitudes and interpolate
    ppg_peaks_amp = np.abs(ppg_df["PPG_Clean"].iloc[ppg_peaks_loc])
    ppg_amp = nk.signal_interpolate(
        ppg_peaks_loc,
        ppg_peaks_amp,
        np.arange(ppg_df.shape[0]),
        method="monotone_cubic",
    )
    return {
        "heart_rate": ppg_df["PPG_Rate"].to_numpy(),
        "ppg_amplitude": np.asarray(ppg_amp),
    }


def extract_resp_features(ts: np.ndarray, sf: float) -> dict[str, np.ndarray]:
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
        sampling_rate=sf,  # type: ignore
    )
    return {
        "resp_amp": resp_features["RSP_Amplitude"].to_numpy(),
        "resp_rate": resp_features["RSP_Rate"].to_numpy(),
        "resp_rvt": resp_features["RSP_RVT"].to_numpy(),
    }


def extract_resp_co2_features(
    ts: np.ndarray,
    sf: float,
) -> dict[str, np.ndarray]:
    """
    Extract end-tidal CO2 waveforms from raw
    CO2 recordings through peak detection and interpolation. Note,
    end-tidal CO2 is extracted from the maxima of the CO2 signal.

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
        sampling_rate=sf,  # type: ignore
        lowcut=0.1,
        highcut=0.4,
        method="butterworth",
        order=4,
    )
    # ensure signal is numpy array
    ts_filt = np.asarray(ts_filt)
    # find peaks (end-tidal CO2 points)
    co2_peaks, peaks_info = find_peaks(
        ts_filt,
        height=np.percentile(ts_filt, 50),  # only consider peaks above 50th percentile
        distance=sf * 1.5,  # minimum distance of 1.5 seconds between peaks
    )
    endtidal_co2 = nk.signal_interpolate(
        co2_peaks,
        peaks_info["peak_heights"],
        np.arange(len(ts_filt)),
        method="monotone_cubic",
    )
    return {"endtidal_co2": endtidal_co2}


def extract_resp_o2_features(
    ts: np.ndarray,
    sf: float,
) -> dict[str, np.ndarray]:
    """
    Extract end-tidal O2 waveforms from raw
    O2 recordings through peak detection and interpolation. Note,
    end-tidal O2 is extracted from the minima of the O2 signal, as
    opposed to the maxima for CO2.

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
        sampling_rate=sf,  # type: ignore
        lowcut=0.1,
        highcut=0.4,
        method="butterworth",
        order=4,
    )
    # ensure signal is numpy array
    ts_filt = np.asarray(ts_filt)
    # invert signal to find minima as peaks
    ts_filt_neg = np.array(ts_filt) * -1
    # find minima (end-tidal O2 points)
    o2_peaks, peaks_info = find_peaks(
        ts_filt_neg,
        height=np.percentile(
            ts_filt_neg, 50
        ),  # only consider peaks above 50th percentile
        distance=sf * 1.5,  # minimum distance of 1.5 seconds between peaks
    )
    # the O2 signal has a prominent transient artifact at the first peak of the recording
    # so we will set the first peak equal to the second peak
    if len(o2_peaks) > 1:
        peaks_info["peak_heights"][0] = peaks_info["peak_heights"][1]

    endtidal_o2 = nk.signal_interpolate(
        o2_peaks,
        peaks_info["peak_heights"] * -1,
        np.arange(len(ts_filt)),
        method="monotone_cubic",
    )

    return {"endtidal_o2": endtidal_o2}


def _trim_physio_signals_dummy(
    physio_df: pd.DataFrame, dummy_n_vols: int, fmri_tr: float, physio_sf: float
) -> pd.DataFrame:
    """
    Trim physiological signals to match the removal of dummy volumes in fMRI data.

    Parameters
    ----------
    physio_df : pd.DataFrame
        The physiological data.
    dummy_n_vols : int
        The number of dummy volumes to remove.
    tr : float
        The repetition time (TR) of the fMRI data.

    Returns
    -------
    pd.DataFrame
        The trimmed physiological data.
    """
    # calculate the time to trim from the start
    trim_time = dummy_n_vols * fmri_tr
    # calculate the number of samples to trim based on the sampling frequency
    trim_n_samples = int(trim_time * physio_sf)
    # trim the physiological data
    physio_df = physio_df.iloc[trim_n_samples:, :].copy()
    return physio_df


def _physio_resample_to_fmri(
    physio_ts: np.ndarray,
    physio_sf: float,
    fmri_n_tp: int,
    fmri_tr: float,
    slicetiming_ref: float,
) -> np.ndarray:
    """
    Resample physiological time series to match fMRI time points.

    Parameters
    ----------
    physio_ts : np.ndarray
        The physiological time series.
    physio_sf : float
        The sampling frequency of the physiological data.
    fmri_n_tp : int
        The number of time points in the fMRI data.
    fmri_tr : float
        The repetition time (TR) of the fMRI data.

    Returns
    -------
    np.ndarray
        The resampled physiological time series.
    """
    # calculate the time points of the fMRI data
    # define sample time points (fmri_times)
    fmri_times = np.arange(0, fmri_n_tp * fmri_tr, fmri_tr)
    fmri_times += fmri_tr * slicetiming_ref  # middle of the TR
    # define physio time points
    physio_times = np.arange(0, len(physio_ts)) / physio_sf
    # interpolate physio to fmri time points
    physio_ts_resamp = nk.signal_interpolate(
        x_values=physio_times,
        y_values=physio_ts,
        x_new=fmri_times,
        method="monotone_cubic",
    )

    return physio_ts_resamp


def _physio_write_image(fp_out: str, ts: np.ndarray, sf: float, label: str) -> None:
    """
    Write physiological time series to image file.

    Parameters
    ----------
    fp_out : str
        The file path to save the image.
    ts : np.ndarray
        The physiological time series.
    sf : float
        The sampling frequency of the physiological data.
    label : str
        The label for the physiological signal.

    Returns
    -------
    None
    """
    # plot the time series and save to file
    fig, ax = plt.subplots(figsize=(15, 5))
    signal_t = np.arange(len(ts)) * (1 / sf)
    ax.plot(signal_t, ts, label=label)
    ax.legend()
    plt.savefig(fp_out)
    plt.close()


def _load_physio_euskalibur(
    physio_fp: str, physio_json: str, physio_cols: List[str]
) -> Tuple[pd.DataFrame, float]:
    """
    Load physiological data from Euskalibur dataset.

    Parameters
    ----------
    physio_fp : str
        The file path to the physiological data.
    physio_json : str
        The file path to the physiological JSON sidecar.
    physio_cols : List[str]
        The list of physiological columns to load.

    Returns
    -------
    pd.DataFrame
        The physiological data.
    float
        the sampling frequency of the physiological data.
    """
    # load physio df
    physio_df = pd.read_csv(physio_fp, sep="\t", compression="gzip", header=None)
    # load physio json
    with open(physio_json, "r") as f:
        physio_json_data = json.load(f)
    # set columns from json sidecar
    if "Columns" in physio_json_data:
        physio_df.columns = physio_json_data["Columns"]
    else:
        raise ValueError("Columns not found in JSON sidecar.")
    # get the sampling frequency from json sidecar
    if "SamplingFrequency" in physio_json_data:
        sampling_freq = physio_json_data["SamplingFrequency"]
    else:
        raise ValueError("SamplingFrequency not found in JSON sidecar.")

    # trim physio signals to start and end from first trigger
    physio_df = _trim_physio_signals_trigger(physio_df)
    return physio_df[physio_cols].copy(), sampling_freq


def _load_physio_nsd(physio_fp: Tuple[str, str]) -> Tuple[pd.DataFrame, float]:
    """
    Load physiological data from NSD dataset.

    Physiological data for pulse and respiration are provided in separate .tsv files, and
    there is no JSON sidecar.

    According to NSD Data Description, the sampling frequency of the physiological data is 50Hz. Also,
    the physiological recordings have already been trimmed to the start and end of the scan triggers, so
    no additional trimming is needed.
    # https://cvnlab.slite.page/p/vjWTghPTb3/Time-series-data


    Parameters
    ----------
    physio_fp : Tuple[str, str]
        The file paths to the physiological data for pulse and respiration. The order
        should be pulse file first, then respiration file.

    Returns
    -------
    pd.DataFrame
        The physiological data.
    float
        the sampling frequency of the physiological data.
    """
    sampling_freq = 50.0
    # load physio df
    physio_df = pd.concat(
        [pd.read_csv(fp, sep="\t", header=None) for fp in physio_fp], axis=1
    )
    # set column names
    physio_df.columns = ["cardiac", "respiratory_effort"]
    return physio_df, sampling_freq


def _trim_physio_signals_trigger(
    physio_df: pd.DataFrame, trigger_col: str = "trigger"
) -> pd.DataFrame:
    """
    Trim physiological signals to start and end of the scan triggers.

    Parameters
    ----------
    physio_df : pd.DataFrame
        The physiological data.
    trigger_col : str, optional
        The name of the trigger column. Defaults to 'trigger'.

    Returns
    -------
    pd.DataFrame
        The trimmed physiological data.
    """
    # find start and end of the scan triggers (round to nearest int)
    trigger = physio_df[trigger_col].round().astype(int)
    # find indices of trigger value 5
    trigger_indx = np.where(trigger == 5)[0]
    # if no triggers found, raise error
    if len(trigger_indx) == 0:
        raise ValueError("No triggers found in physiological data.")
    # get first and last trigger index
    first_trigger = trigger_indx[0]
    last_trigger = trigger_indx[-1]
    # trim physio to start of trigger
    physio_df = physio_df.iloc[first_trigger:last_trigger, :].copy()
    return physio_df
