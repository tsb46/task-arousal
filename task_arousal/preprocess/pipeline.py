"""
Preprocessing pipeline for fMRI and physiological data.

fMRI preprocessing is performed on outputs from fMRIPrep, including:

1) Drop dummy volumes
2) Detrending
3) Band-pass filtering (0.01 - 0.25 Hz)
4) Standardize signal (z-score)
5) Smoothing

Physio preprocessing is performed on raw physiological data, including:

1) Feature extraction
2) Band-pass filtering
3) Resampling to fMRI time points
"""
import json
import os

from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import neurokit2 as nk
import numpy as np
import pandas as pd

from nilearn.image import clean_img, smooth_img
from nilearn.masking import apply_mask, unmask
from scipy.stats import zscore

from task_arousal.constants import MASK, TR, SLICE_TIMING_REF
from task_arousal.io.file import FileMapper
from task_arousal.preprocess.physio_features import (
    extract_ppg_features, 
    extract_resp_features,
    extract_resp_co2_features,
    extract_resp_o2_features
)

## Preprocessing parameters
# Number of dummy volumes to drop 
DUMMY_VOLUMES = 10
# High-pass filter cutoff frequency for fmri
HIGHPASS = 0.01
# Low-pass filter cutoff frequency for fmri
LOWPASS = 0.25
# Full width at half maximum for Gaussian smoothing
FWHM = 4  # in mm
# physio fields to extract from raw data
PHYSIO_COLUMNS = ['respiratory_effort', 'cardiac', 'respiratory_CO2', 'respiratory_O2']
# physiological resample frequency (in Hz)
PHYSIO_RESAMPLE_F = 50


class PreprocessingPipeline:
    """
    Preprocessing pipeline for fMRI and physiological data.
    """
    def __init__(self, subject: str):
        """Initialize the preprocessing pipeline.

        Parameters
        ----------
            subject (str): The subject identifier.
        """
        self.subject = subject
        # map file paths associated to subject
        self.file_mapper = FileMapper(subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks

    def preprocess(
        self, 
        task: str | None = None,
        sessions: list[str] | None = None,
        skip_func: bool = False,
        skip_physio: bool = False,
        save_physio_figs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Preprocess fMRI and physiological data.

        Parameters
        ----------
        task : str or None, optional
            The task identifier. If no task identifier is provided, all tasks will be processed. Defaults to None.
        sessions : list of str or None, optional
            The session identifiers. If no session identifiers are provided, all sessions will be processed. Defaults to None.
        skip_func : bool, optional
            Whether to skip fMRI preprocessing. Defaults to False.
        skip_physio : bool, optional
            Whether to skip physiological preprocessing. Defaults to False.
        """
        if skip_func & skip_physio:
            raise ValueError("Both fMRI and physiological preprocessing cannot be skipped.")
        if skip_func & verbose:
            print(f"Skipping fMRI preprocessing for subject '{self.subject}'...")
        if skip_physio & verbose:
            print(f"Skipping physiological preprocessing for subject '{self.subject}'...")

        # if task is not None, ensure it's an available task
        if task is not None:
            if task not in self.tasks:
                raise ValueError(
                    f"Task '{task}' is not available for subject '{self.subject}'."
                )
            tasks_to_process = [task]
        else:
            tasks_to_process = self.tasks
        if verbose:
            print(f"Processing tasks for subject '{self.subject}': {tasks_to_process}")

        # loop through each task and process scans and physio from all sessions (and runs)
        for task in tasks_to_process:
            if verbose:
                print(
                    f"Processing task '{task}' for subject '{self.subject}' "
                    f"and sessions '{sessions if sessions is not None else 'all'}'..."
                )

            # functional MRI preprocessing
            if not skip_func:
                # get fmri files for task
                fmri_files = self.file_mapper.get_fmri_files(task, sessions=sessions)
                # loop through fmri files and preprocess
                for fmri_file in fmri_files:
                    if verbose:
                        print(f"Preprocessing fMRI file: {fmri_file}")
                    # Apply the functional MRI preprocessing pipeline
                    fmri_proc = func_pipeline(fmri_file)
                    # Write out the preprocessed fMRI file
                    self.write_out_fmri_file(fmri_proc, fmri_file)

            # physio preprocessing pipeline
            if not skip_physio:
                # get physio files (and JSON sidecars) for task
                physio_files = self.file_mapper.get_physio_files(task, return_json=True, sessions=sessions)
                # loop through physio files and preprocess
                for physio_file in physio_files:
                    if verbose:
                        print(f"Preprocessing physiological file: {physio_file[0]}")

                    # find matching fmri file to get number of volumes for resampling
                    file_entities = self.file_mapper.layout.parse_file_entities(physio_file[0])
                    matching_fmri_files = self.file_mapper.get_matching_files(file_entities, 'fmri')
                    if len(matching_fmri_files) == 0:
                        # in some scenarios, physio may be recorded but no usable fMRI data
                        if verbose:
                            print(f"No matching fMRI file found for physiological file '{physio_file[0]}'.")
                        continue
                    
                    elif len(matching_fmri_files) > 1:
                        raise ValueError(
                            f"Multiple matching fMRI files found for physiological file '{physio_file[0]}'. "
                        )
                    else:
                        matching_fmri_file = matching_fmri_files[0]
                        if verbose:
                            print(f"Found matching fMRI file: {matching_fmri_file}")

                    # Apply the physiological preprocessing pipeline
                    physio_proc = physio_pipeline(
                        physio_fp=physio_file[0], 
                        physio_json=physio_file[1],  
                        fmri_fp=matching_fmri_file, 
                        save_physio_figs=save_physio_figs
                    )
                    # Write out the preprocessed physiological file
                    self.write_out_physio_file(physio_proc, physio_file[0])

            if verbose:
                print(f"Finished processing task '{task}' for subject '{self.subject}'.")

    def write_out_fmri_file(self, fmri_img: nib.Nifti1Image, file_orig: str) -> None: # type: ignore
        """Write out the preprocessed fMRI file.

        Parameters
        ----------
        fmri_img : nib.Nifti1Image
            The preprocessed fMRI image.
        file_orig : str
            The original file path
        """
        # get output directory from original file path
        output_dir = self.file_mapper.get_out_directory(file_orig)
        # get file name from original file path
        file_orig_name = os.path.basename(file_orig)
        # strip 'preproc_bold.nii.gz' from file name
        file_new_name = file_orig_name.rstrip('preproc_bold.nii.gz')
        # add 'preprocfinal_bold.nii.gz' to file name
        file_new = f"{file_new_name}preprocfinal_bold.nii.gz"
        output_path = f"{output_dir}/{file_new}"
        nib.save(fmri_img, output_path) # type: ignore
    
    def write_out_physio_file(self, physio_dict: dict[str, np.ndarray], file_orig: str) -> None:
        """Write out the preprocessed physiological file.

        Parameters
        ----------
        physio_dict : dict[str, np.ndarray]
            The preprocessed physiological data.
        file_orig : str
            The original file path
        """
        # convert to dataframe
        physio_df = pd.DataFrame(physio_dict)
        # get output directory from original file path
        output_dir = self.file_mapper.get_out_directory(file_orig)
        # get file name from original file path
        file_orig_name = os.path.basename(file_orig)
        # strip 'physio.tsv.gz' from file name
        file_new_name = file_orig_name.rstrip('physio.tsv.gz')
        # add 'desc-preproc_physio.tsv.gz' to file name
        file_new = f"{file_new_name}desc-preproc_physio.tsv.gz"
        # write out as tsv.gz with pandas
        physio_df.to_csv(
            f"{output_dir}/{file_new}", 
            sep="\t", 
            index=False, 
            compression='gzip'
        )


def func_pipeline(func_fp: str) -> nib.Nifti1Image: # type: ignore
    """
    Function pipeline for processing functional MRI data.

    Preprocessing steps:

    1) Drop dummy volumes
    2) Detrending
    3) Band-pass filtering
    4) Standardization
    5) Smoothing

    Parameters
    ----------
    func_fp : str
        The file path to the functional MRI data.

    Returns
    -------
    nb.Nifti1Image
        The processed functional MRI data.
    """
    # Load functional MRI data
    func_img = nib.load(func_fp) # type: ignore

    # drop dummy volumes
    func_img_proc = _func_trim(func_img, DUMMY_VOLUMES) # type: ignore

    # using the clean_img function to detrend, high-pass filter, and standardize the signal
    func_img_proc = clean_img(
        func_img_proc, 
        detrend=True, 
        standardize=True,
        high_pass=HIGHPASS,
        low_pass=LOWPASS,
        mask_img=MASK,
        t_r=TR
    )

    # Apply spatial smoothing
    func_img_proc = _func_smooth(func_img_proc, fwhm=FWHM) # type: ignore

    # Mask out smoothed data to ensure non-brain voxels are zero
    mask_img = nib.load(MASK) # type: ignore
    func_data_masked = apply_mask(func_img_proc, mask_img)
    func_img_proc = unmask(func_data_masked, mask_img)
    
    return func_img_proc # type: ignore


def physio_pipeline(
    physio_fp: str, 
    physio_json: str, 
    fmri_fp: str, 
    save_physio_figs: bool = False
) -> dict[str, np.ndarray]:
    """
    Physiological pipeline for processing physiological data.

    Preprocessing steps:

    1) Resample to 50Hz (polyphase filtering to avoid aliasing)
    2) Feature extraction
    3) Resample to fMRI time points (low-pass filter to avoid aliasing, then interpolate)
    4) Band-pass filtering to match fMRI bandpass (0.01 - 0.25 Hz)
    5) Standardization

    Parameters
    ----------
    physio_fp : str
        The file path to the physiological data.
    physio_json : str
        The file path to the physiological JSON sidecar.
    fmri_fp : str
        The file path to the functional MRI data (to get number of time points for resampling).
    save_physio_figs : bool, optional
        Whether to save figures of the physiological signals, by default False.

    Returns
    -------
    dict[str, np.ndarray]
        The processed physiological data.
    """
    # Load physiological data
    physio_dict, physio_sf = _load_physio(physio_fp, physio_json)
    # load fmri data
    fmri_img = nib.load(fmri_fp) # type: ignore
    fmri_n_tp = fmri_img.shape[-1] - DUMMY_VOLUMES # type: ignore
    # define function for feature extraction based on column name
    feature_extraction_funcs = {
        'respiratory_effort': extract_resp_features,
        'cardiac': extract_ppg_features,
        'respiratory_CO2': extract_resp_co2_features,
        'respiratory_O2': extract_resp_o2_features,
    }
    # loop through physio columns and extract features
    physio_data_proc = {}
    for col in PHYSIO_COLUMNS:
        # physio has high sampling rate (1000Hz), so we can downsample to 50Hz
        # use polyphase filtering to avoid aliasing
        physio_resampled = nk.signal_resample(
            physio_dict[col], 
            sampling_rate=physio_sf,
            desired_sampling_rate=PHYSIO_RESAMPLE_F,
            method='poly'
        )
        if save_physio_figs:
            # save figure of resampled physio signal
            fig_fp = physio_fp.replace('.tsv.gz', f'_{col}.png')
            _physio_write_image(
                fp_out=fig_fp, 
                ts=physio_resampled, # type: ignore
                sf=PHYSIO_RESAMPLE_F, 
                label=col
            )
        assert isinstance(physio_resampled, np.ndarray), "Resampled physiological data is not a numpy array."
        # extract features
        if col in feature_extraction_funcs:
            physio_features = feature_extraction_funcs[col](physio_resampled, PHYSIO_RESAMPLE_F)
            # loop through features and resample to fMRI time points
            for feat_name, feat_ts in physio_features.items():
                # first, low-pass filter the time series to avoid aliasing
                feat_ts_lowpass = nk.signal_filter(
                    feat_ts,
                    sampling_rate=PHYSIO_RESAMPLE_F,
                    highcut=1/(2*TR), # Nyquist frequency
                    method='butterworth',
                    order=4
                )
                # next, interpolate to fMRI time points
                feat_ts_resampled = _physio_resample_to_fmri(
                    feat_ts_lowpass, 
                    physio_sf=PHYSIO_RESAMPLE_F, 
                    fmri_n_tp=fmri_n_tp, 
                    fmri_tr=TR,
                    slicetiming_ref=SLICE_TIMING_REF
                )
                # finally, band-pass filter to match fMRI bandpass
                feat_ts_bandpassed = nk.signal_filter(
                    feat_ts_resampled,
                    sampling_rate=1/TR, # type: ignore
                    lowcut=HIGHPASS,
                    highcut=LOWPASS,
                    method='butterworth',
                    order=4
                )
                # standardize the signal (z-score)
                feat_ts_bandpassed = zscore(feat_ts_bandpassed)
                # add to processed physio dict
                physio_data_proc[f"{feat_name}"] = feat_ts_bandpassed
        else:
            print(f"No feature extraction function defined for column '{col}'. Skipping.")
    
    return physio_data_proc


def _func_trim(func_img: nib.Nifti1Image, start: int) -> nib.Nifti1Image: # type: ignore
    """
    Trim the functional MRI data.

    Parameters
    ----------
    func_img : nib.Nifti1Image
        The functional MRI data.
    start : int
        The start index for trimming.

    Returns
    -------
    nib.Nifti1Image
        The trimmed functional MRI data.
    """
    # Get the data from the NIfTI image
    data = func_img.get_fdata()
    # Trim the data
    trimmed_data = data[..., start:]
    # Create a new NIfTI image with the trimmed data
    trimmed_img = nib.Nifti1Image(trimmed_data, func_img.affine, func_img.header) # type: ignore
    return trimmed_img


def _func_smooth(func_img: nib.Nifti1Image, fwhm: float) -> nib.Nifti1Image: # type: ignore
    """
    Apply smoothing to functional MRI data.

    Parameters
    ----------
    func_img : nib.Nifti1Image
        The functional MRI data.
    fwhm : float
        The full width at half maximum (FWHM) for the Gaussian smoothing kernel.

    Returns
    -------
    nib.Nifti1Image
        The smoothed functional MRI data.
    """
    # Apply smoothing (e.g., using a Gaussian filter)
    smoothed_img = smooth_img(func_img, fwhm=fwhm)
    return smoothed_img # type: ignore


def _load_physio(physio_fp: str, physio_json: str) -> Tuple[dict[str, list[float]], float]:
    """
    Load physiological data from a file, and trim to start and end from the first trigger.

    Parameters
    ----------
    physio_fp : str
        The file path to the physiological data.
    physio_json : str
        The file path to the physiological JSON sidecar.

    Returns
    -------
    dict[str, float]
        A dictionary containing each physiological label as a key 
        and its corresponding time series as a list of floats.
    float
        the sampling frequency of the physiological data.
    """
    # load physio df
    physio_df = pd.read_csv(physio_fp, sep="\t", compression='gzip', header=None)
    # load physio json
    with open(physio_json, 'r') as f:
        physio_json_data = json.load(f)
    # set columns from json sidecar
    if 'Columns' in physio_json_data:
        physio_df.columns = physio_json_data['Columns']
    else:
        raise ValueError("Columns not found in JSON sidecar.")
    # get the sampling frequency from json sidecar
    if 'SamplingFrequency' in physio_json_data:
        sampling_freq = physio_json_data['SamplingFrequency']
    else:
        raise ValueError("SamplingFrequency not found in JSON sidecar.")
    # trim physio signals to start and end from first trigger
    physio_df = _trim_physio_signals_trigger(physio_df)
    # trim physio signals to match removal of fMRI dummy volumes
    physio_df = _trim_physio_signals_dummy(
        physio_df, DUMMY_VOLUMES, TR, sampling_freq
    )
    # convert to dict
    try:
        physio_dict = physio_df[PHYSIO_COLUMNS].to_dict(orient='list')
    except KeyError as e:
        raise ValueError(f"Missing expected column in physiological data: {e}")
    return physio_dict, sampling_freq # type: ignore


def _trim_physio_signals_trigger(
    physio_df: pd.DataFrame, 
    trigger_col: str ='trigger'
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


def _trim_physio_signals_dummy(
    physio_df: pd.DataFrame, 
    dummy_n_vols: int,
    fmri_tr: float,
    physio_sf: float
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
    slicetiming_ref: float
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
    fmri_times = np.arange(0, fmri_n_tp*fmri_tr, fmri_tr)
    fmri_times += (fmri_tr * slicetiming_ref)  # middle of the TR
    # define physio time points
    physio_times = np.arange(0, len(physio_ts))/physio_sf
    # interpolate physio to fmri time points
    physio_ts_resamp = nk.signal_interpolate(
        x_values=physio_times, 
        y_values=physio_ts,
        x_new=fmri_times, 
        method='monotone_cubic'
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
    fig, ax = plt.subplots(figsize=(15,5))
    signal_t = np.arange(len(ts))*(1/sf)
    ax.plot(signal_t, ts, label=label)
    ax.legend()
    plt.savefig(fp_out)
    plt.close()