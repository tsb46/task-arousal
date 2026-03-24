"""
Volume functional MRI preprocessing component.
"""

from __future__ import annotations

import subprocess
import tempfile

from pathlib import Path

import nibabel as nib
import numpy as np

from nilearn.image import clean_img, smooth_img
from nilearn.masking import apply_mask, unmask


def func_volume_pipeline(
    func_fp: str,
    tr: float,
    brain_mask_fp: str,
    fwhm: float,
    dummy_vols: int,
    highpass: float | None,
    remove_dummy: bool = True,
    spatial_smooth: bool = True,
    detrend: bool = True,
    highpass_filter: bool = True,
    standardize: bool = True,
    to_std: bool = False,
    native_to_t1w_fp: str | None = None,
    t1w_to_std_fp: str | None = None,
    std_space_ref_fp: str | None = None,
) -> nib.nifti1.Nifti1Image:
    """
    Functional volume pipeline for processing functional MRI data.

    Preprocessing steps:

    (Perform transforms to standard space if to_std is True, using func_to_std)

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
    highpass : float | None
        The high-pass filter cutoff frequency in Hz. If highpass_filter is False,
        this parameter is ignored and no high-pass filtering is applied.
    resample : bool, optional
        Whether to resample the fMRI data to the brain mask resolution, by default False.
    remove_dummy : bool, optional
        Whether to remove dummy volumes, by default True.
    spatial_smooth : bool, optional
        Whether to apply spatial smoothing, by default True.
    detrend : bool, optional
        Whether to apply detrending, by default True.
    highpass_filter : bool, optional
        Whether to apply high-pass filtering, by default True.
    standardize : bool, optional
        Whether to apply standardization, by default True.

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
    if highpass is not None and highpass < 0:
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

    # if to_std is True, apply the standard space transformations to the functional image before any other preprocessing steps
    # using the provided ANTs transformation files from fMRIPrep (native_to_t1w_fp and t1w_to_std_fp) and the standard space reference image (std_space_ref_fp)
    if to_std:
        if (
            native_to_t1w_fp is None
            or t1w_to_std_fp is None
            or std_space_ref_fp is None
        ):
            raise ValueError(
                "native_to_t1w_fp, t1w_to_std_fp, and std_space_ref_fp must all be provided if to_std is True"
            )
        func_img = func_to_std(
            img=func_img,
            std_space_ref_fp=std_space_ref_fp,
            native_to_t1w_fp=native_to_t1w_fp,
            t1w_to_std_fp=t1w_to_std_fp,
        )
    # Make sure func and mask grids match in XYZ.
    if func_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            "Functional image and mask have different spatial shapes. "
            f"func shape[:3]={func_img.shape[:3]} vs mask shape[:3]={mask_img.shape[:3]}. "
            "Provide a mask in the same space/resolution as func_fp."
        )

    if remove_dummy:
        n_tp = int(func_img.shape[3])
        if dummy_vols >= n_tp:
            raise ValueError(
                f"dummy_vols ({dummy_vols}) must be < number of timepoints ({n_tp})"
            )
        func_img_proc = _func_trim(func_img, dummy_vols)
    else:
        func_img_proc = func_img

    # if highpass_filter is False, set highpass to None to skip high-pass filtering in clean_img
    if not highpass_filter:
        highpass = None
    # using the clean_img function to detrend, high-pass filter, and standardize the signal
    func_img_proc = clean_img(
        func_img_proc,
        detrend=detrend,
        standardize=standardize,
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


def func_to_std(
    img: str | nib.nifti1.Nifti1Image,
    std_space_ref_fp: str,
    native_to_t1w_fp: str,
    t1w_to_std_fp: str,
    output_fp: str | None = None,
) -> nib.nifti1.Nifti1Image:
    """
    Apply the standard fMRIPrep spatial transformations to the given image, to bring it into MNI space.

    Inspired from:
    https://tedana.readthedocs.io/en/stable/faq.html#warping-scanner-space-fmriprep-outputs-to-standard-space

    Parameters
    ----------
    img : str or nib.Nifti1Image
        Image to transform. If a NIfTI image is provided, it will be written to a
        temporary directory before calling ANTs.
    std_space_ref_fp : str
        File path to the standard space reference image (e.g. MNI152NLin2009cAsym).
    native_to_t1w_fp : str
        File path to the fMRIPrep-generated transformation file from native space to T1w space (e.g. from-boldref_to-T1w_mode-image_xfm.txt).
    t1w_to_std_fp : str
        File path to the fMRIPrep-generated transformation file from T1w space to standard space (e.g. from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5).
    output_fp : str or None
        File path where the transformed image should be saved. If None, a
        temporary output path is used and the transformed image is returned in memory.

    Returns
    -------
    nib.Nifti1Image
        The transformed image.

    """

    def _load_materialized_nifti(img_fp: str) -> nib.nifti1.Nifti1Image:
        loaded_img = nib.nifti1.load(img_fp)
        if not isinstance(loaded_img, nib.nifti1.Nifti1Image):
            raise TypeError(f"Expected NIfTI image at {img_fp}, got {type(loaded_img)}")
        return nib.nifti1.Nifti1Image(
            np.asanyarray(loaded_img.dataobj),
            affine=loaded_img.affine,
            header=loaded_img.header.copy(),
            extra=loaded_img.extra.copy(),
        )

    with tempfile.TemporaryDirectory(prefix="task_arousal_multiecho_") as tmpdir:
        if isinstance(img, nib.nifti1.Nifti1Image):
            img_fp = f"{tmpdir}/multiecho_native.nii.gz"
            nib.nifti1.save(img, img_fp)
        else:
            img_fp = img

        resolved_output_fp = output_fp or f"{tmpdir}/multiecho_std.nii.gz"

        try:
            subprocess.run(
                [
                    "antsApplyTransforms",
                    "-e",
                    "3",
                    "-i",
                    img_fp,
                    "-r",
                    std_space_ref_fp,
                    "-o",
                    resolved_output_fp,
                    "-n",
                    "LanczosWindowedSinc",
                    "-t",
                    t1w_to_std_fp,
                    "-t",
                    native_to_t1w_fp,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "antsApplyTransforms failed while transforming multi-echo image: "
                f"{exc.stderr.strip() or exc.stdout.strip() or exc}"
            ) from exc

        if output_fp is None:
            return _load_materialized_nifti(resolved_output_fp)

    saved_img = nib.nifti1.load(output_fp)
    if not isinstance(saved_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Expected NIfTI image at {output_fp}, got {type(saved_img)}")
    return saved_img
