"""
Volume functional MRI preprocessing component.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib

from nilearn.image import clean_img, smooth_img, resample_img
from nilearn.masking import apply_mask, unmask


def func_volume_pipeline(
    func_fp: str,
    tr: float,
    brain_mask_fp: str,
    fwhm: float,
    dummy_vols: int,
    highpass: float,
    resample: bool = False,
    remove_dummy: bool = True,
) -> nib.nifti1.Nifti1Image:
    """
    Functional volume pipeline for processing functional MRI data.

    Preprocessing steps:

    1) Drop dummy volumes
    2) Detrending (clean_img)
    3) High-pass filtering (> 0.01 Hz) (clean_img)
    4) Standardization (clean_img)
    5) Smoothing (smooth_img)

    Parameters
    ----------
    func_fp : str
        The file path to the functional MRI data.
    tr : float
        The repetition time (TR) of the fMRI data.
    brain_mask_fp : str
        The file path to the brain mask.
    fwhm : float
        The full width at half maximum (FWHM) for spatial smoothing.
    dummy_vols : int
        The number of dummy volumes to drop. Ignored if remove_dummy is False.
    highpass : float
        The high-pass filter cutoff frequency in Hz.
    resample : bool, optional
        Whether to resample the fMRI data to the brain mask resolution, by default False.
    remove_dummy : bool, optional
        Whether to remove dummy volumes, by default True.

    Returns
    -------
    nib.Nifti1Image
        The processed functional MRI data.
    """

    func_fp_p = Path(func_fp)
    if not func_fp_p.exists():
        raise FileNotFoundError(f"Functional file not found: {func_fp}")

    mask_fp_p = Path(brain_mask_fp)
    if not mask_fp_p.exists():
        raise FileNotFoundError(f"Brain mask file not found: {brain_mask_fp}")

    if tr <= 0:
        raise ValueError(f"tr must be > 0, got {tr}")
    if highpass < 0:
        raise ValueError(f"highpass must be >= 0, got {highpass}")
    if dummy_vols < 0:
        raise ValueError(f"dummy_vols must be >= 0, got {dummy_vols}")
    if fwhm is None:
        raise ValueError("fwhm must not be None")

    # Load functional MRI data
    func_img = nib.nifti1.load(func_fp)

    # load mask
    mask_img = nib.nifti1.load(brain_mask_fp)

    # ensure correct types and dimensionalities
    if not isinstance(func_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Loaded fMRI data is not a Nifti1Image: {type(func_img)}")
    if not isinstance(mask_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Loaded mask is not a Nifti1Image: {type(mask_img)}")
    if mask_img.ndim != 3:
        raise ValueError(f"brain_mask_fp must be 3D, got shape {mask_img.shape}")
    if func_img.ndim != 4:
        raise ValueError(f"func_fp must be 4D (x,y,z,t), got shape {func_img.shape}")

    # If not resampling, make sure grids match in XYZ.
    if not resample and func_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            "Functional image and mask have different spatial shapes. "
            f"func shape[:3]={func_img.shape[:3]} vs mask shape[:3]={mask_img.shape[:3]}. "
            "Set resample=True or provide a mask in the same space/resolution as func_fp."
        )

    # downsample data to mask resolution, assumes func is in same space as mask
    if resample:
        func_img = resample_img(
            func_img,
            target_affine=mask_img.affine,
            target_shape=mask_img.shape[:3],
            interpolation="continuous",
            copy_header=True,
            force_resample=True,
        )

        if not isinstance(func_img, nib.nifti1.Nifti1Image):
            raise TypeError("resample_img did not return a Nifti1Image")

    if remove_dummy:
        n_tp = int(func_img.shape[3])
        if dummy_vols >= n_tp:
            raise ValueError(
                f"dummy_vols ({dummy_vols}) must be < number of timepoints ({n_tp})"
            )
        func_img_proc = _func_trim(func_img, dummy_vols)
    else:
        func_img_proc = func_img

    # using the clean_img function to detrend, high-pass filter, and standardize the signal
    func_img_proc = clean_img(
        func_img_proc,
        detrend=True,
        standardize=True,
        high_pass=highpass,
        mask_img=mask_img,
        t_r=tr,
    )
    # ensure nifti after clean_img
    assert isinstance(func_img_proc, nib.nifti1.Nifti1Image), (
        "clean_img did not return a Nifti1Image."
    )

    # Apply spatial smoothing
    if float(fwhm) > 0:
        func_img_proc = _func_smooth(func_img_proc, fwhm=float(fwhm))

    # Mask out smoothed data to ensure non-brain voxels are zero
    func_data_masked = apply_mask(func_img_proc, mask_img)
    func_img_proc = unmask(func_data_masked, mask_img)

    # ensure nifti after unmask
    assert isinstance(func_img_proc, nib.nifti1.Nifti1Image), (
        "unmask did not return a Nifti1Image."
    )

    return func_img_proc


def _func_trim(func_img: nib.Nifti1Image, start: int) -> nib.Nifti1Image:  # type: ignore
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
    trimmed_img = nib.Nifti1Image(trimmed_data, func_img.affine, func_img.header)  # type: ignore
    return trimmed_img


def _func_smooth(func_img: nib.Nifti1Image, fwhm: float) -> nib.Nifti1Image:  # type: ignore
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
    return smoothed_img  # type: ignore
