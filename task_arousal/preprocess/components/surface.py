"""
Surface-based volume fMRI preprocessing components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from nibabel.cifti2.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import SeriesAxis
from nilearn.signal import clean

from .workbench import cifti_smooth


def func_surface_pipeline(
    func_fp: str,
    tr: float,
    dummy_vols: int,
    highpass: float,
    fwhm: float,
    surface_template_lh: str,
    surface_template_rh: str,
    remove_dummy: bool = True,
):
    """
    Surface volume pipeline for processing functional MRI data.

    Preprocessing steps:
    1) Drop dummy volumes
    2) Detrending (clean)
    3) High-pass filtering (> 0.01 Hz) (clean)
    4) Standardization (clean)
    5) Smoothing (cifti_smooth)

    Parameters
    ----------
    func_fp : str
        The file path to the functional MRI data.
    tr : float
        The repetition time (TR) of the fMRI data.
    dummy_vols : int
        The number of dummy volumes to drop. Ignored if remove_dummy is False.
    highpass : float
        The high-pass filter cutoff frequency in Hz.
    fwhm : float
        The full width at half maximum (FWHM) for spatial smoothing.
    surface_template_lh : str
        The file path to the left hemisphere surface template.
    surface_template_rh : str
        The file path to the right hemisphere surface template.
    remove_dummy : bool, optional
        Whether to remove dummy volumes, by default True.
    """
    if tr <= 0:
        raise ValueError(f"tr must be > 0, got {tr}")
    if highpass < 0:
        raise ValueError(f"highpass must be >= 0, got {highpass}")
    if dummy_vols < 0:
        raise ValueError(f"dummy_vols must be >= 0, got {dummy_vols}")
    if fwhm is None:
        raise ValueError("fwhm must not be None")

    func_fp_p = Path(func_fp)
    img = Cifti2Image.from_filename(str(func_fp_p))

    data = np.asanyarray(img.dataobj)
    if data.ndim != 2:
        raise ValueError(f"Expected dtseries to be 2D, got shape {data.shape}")

    # Identify time axis (usually axis 0 for dtseries)
    ax0 = img.header.get_axis(0)
    ax1 = img.header.get_axis(1)
    time_axis = (
        0 if isinstance(ax0, SeriesAxis) else 1 if isinstance(ax1, SeriesAxis) else None
    )
    if time_axis is None:
        raise ValueError(
            "Could not identify SeriesAxis (time) in CIFTI header; expected dtseries."
        )

    # Put data into shape (T, V) for nilearn.signal.clean
    if time_axis == 0:
        time_by_feat = data
        series_axis: SeriesAxis = ax0  # type: ignore[assignment]
        other_axis = ax1
    else:
        time_by_feat = data.T
        series_axis = ax1  # type: ignore[assignment]
        other_axis = ax0

    if remove_dummy:
        if dummy_vols < 0:
            raise ValueError("dummy_vols must be >= 0")
        if dummy_vols >= time_by_feat.shape[0]:
            raise ValueError(
                f"dummy_vols ({dummy_vols}) must be < number of timepoints ({time_by_feat.shape[0]})"
            )
        time_by_feat = time_by_feat[dummy_vols:, :]

        # Update the SeriesAxis to reflect the dropped frames
        new_start = float(series_axis.start) + float(dummy_vols) * float(
            series_axis.step
        )
        new_size = int(series_axis.size) - int(dummy_vols)
        series_axis = SeriesAxis(
            start=new_start, step=float(series_axis.step), size=new_size
        )

    # Detrend, high-pass filter, and standardize (z-score) along time
    time_by_feat = clean(
        time_by_feat,
        detrend=True,
        standardize="zscore",
        high_pass=highpass,
        t_r=tr,
    )
    time_by_feat = np.asarray(time_by_feat, dtype=np.float32)

    # Restore original axis order
    if time_axis == 0:
        cleaned_data = time_by_feat
        new_axes = (series_axis, other_axis)
    else:
        cleaned_data = time_by_feat.T
        new_axes = (other_axis, series_axis)

    new_header = Cifti2Header.from_axes(new_axes)
    cleaned_img = Cifti2Image(
        cleaned_data,
        header=new_header,
        nifti_header=img.nifti_header,
    )

    # Surface smoothing via Workbench operates on files, so round-trip through temp files.
    # If fwhm <= 0, skip smoothing.
    if float(fwhm) <= 0:
        return cleaned_img

    with tempfile.TemporaryDirectory(prefix="task_arousal_surface_") as tmpdir:
        tmpdir_p = Path(tmpdir)
        fp_clean = tmpdir_p / "desc-clean.dtseries.nii"
        fp_smooth = tmpdir_p / "desc-clean_smooth.dtseries.nii"
        cleaned_img.to_filename(str(fp_clean))
        cifti_smooth(
            fp_in=str(fp_clean),
            fp_out=str(fp_smooth),
            fwhm=float(fwhm),
            surface_template_lh=surface_template_lh,
            surface_template_rh=surface_template_rh,
        )

        smoothed_img = Cifti2Image.from_filename(str(fp_smooth))
        smoothed_data = np.asarray(smoothed_img.dataobj, dtype=np.float32)
        return Cifti2Image(
            smoothed_data,
            header=smoothed_img.header,
            nifti_header=smoothed_img.nifti_header,
        )
