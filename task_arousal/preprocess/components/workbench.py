# type: ignore
"""Workbench (V1.4.2) command line utilities"""

import nipype.interfaces.workbench as wb


def cifti_smooth(
    fp_in: str,
    fp_out: str,
    fwhm: float,
    surface_template_lh: str,
    surface_template_rh: str,
) -> None:
    """
    Smooth CIFTI file using wb_command -cifti-smoothing

    Parameters
    ----------
        fp_in: str
            filepath to CIFTI file to smooth
        fp_out: str
            filepath to smoothed CIFTI file
        fwhm: float
            FWHM of the Gaussian kernel in mm
        surface_template_lh: str
            filepath to left hemisphere surface template
        surface_template_rh: str
            filepath to right hemisphere surface template
    """
    # template prefix
    template_prefix = "template/fsaverage"

    # convert fwhm to sigma (standard deviation)
    sigma = fwhm / 2.3548
    cifti_smooth = wb.CiftiSmooth()
    cifti_smooth.inputs.in_file = fp_in
    cifti_smooth.inputs.direction = "COLUMN"
    cifti_smooth.inputs.left_surf = surface_template_lh
    cifti_smooth.inputs.right_surf = surface_template_rh
    cifti_smooth.inputs.sigma_surf = sigma
    cifti_smooth.inputs.sigma_vol = sigma
    cifti_smooth.inputs.out_file = fp_out
    cifti_smooth.run()
