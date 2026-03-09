"""
CLI utility that is a wrapper around the transform_data module to transform between
functional, anatomical, MNI and fsaverage spaces provided by the authors:

https://github.com/cvnlab/nsdcode

This only performs a subset of the tranformations provided by the NSD code, specifically:
(1) subject-volume to subject-anatomical space (1.8mm to 1mm)
(2) subject-anatomical to subject-volume space (0.8 mm to 1.8mm)
(3) subject-volume to MNI (FSL) space (1.8mm to MNI)
(4) subject-anatomical to MNI (FSL) space (0.8mm to MNI)
(3) subject-volume to subject native surface space (1.8mm to surface)

"""

import argparse
import os
from typing import Literal

import nibabel as nib

from nsdcode.transform_data import transform_data

# hard code NSD data directory
NSD_DATA_DIR = "data/nsd"


def main(
    subject: str,
    functional_data_path: str | None,
    anatomical_data_path: str | None,
    transform_type: Literal["anat_to_func", "func_to_anat", "func_to_surface"],
    output_dir: str,
    output_prefix: str | None = None,
    interp_type: Literal["nearest", "linear", "cubic"] = "cubic",
):
    """
    Main function to perform transformation between coordinate systems using NSD code.

    Parameters
    ----------
    subject: str
        Subject to perform transformation for. Provide without subject prefix (e.g. '01' for subj01).
    functional_data_path: str or None
        Path to the functional data in subject native space to be transformed. If not provided, the functional data will be loaded from the NSD data directory.
    anatomical_data_path: str or None
        Path to the anatomical data in subject anatomical space to be transformed. If not provided, the anatomical data will be loaded from the NSD data directory.
    transform_type: str
        Type of transformation to perform. One of "anat_to_func", "func_to_anat", "anat_to_mni", "func_to_mni" or "func_to_surface".
    output_dir: str
        Directory to save the transformed data.
    output_prefix: str or None
        Prefix for the output file name. If not provided, the default will be the filename of the input functional or anatomical data with a suffix indicating the transformation type.
    """
    # interp type is always cubic for our use case
    interptype = interp_type

    # must provide anatomical or functional data path depending on the transform type
    if transform_type == "anat_to_func" or transform_type == "anat_to_mni":
        if anatomical_data_path is None:
            raise ValueError(
                "Anatomical data path must be provided for anat_to_func or anat_to_mni transformation."
            )
    elif transform_type == "func_to_anat" or transform_type == "func_to_mni":
        if functional_data_path is None:
            raise ValueError(
                "Functional data path must be provided for func_to_anat or func_to_mni transformation."
            )
    elif transform_type == "func_to_surface":
        if functional_data_path is None:
            raise ValueError(
                "Functional data path must be provided for func_to_surface transformation."
            )

    # determine source and target spaces based on transform_type
    if transform_type == "anat_to_func":
        sourcespace = "anat0pt8"
        targetspace = "func1pt8"
        voxelsize = 1.8
        res = None
        # internal code used by NSD code to identify the correct transform file
        casenum = 1
        # specify transform file for anatomical to functional transformation
        transform_file = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_anat0pt8-to-func1pt8.nii.gz"
        )
        ext = ".nii.gz"
    elif transform_type == "func_to_anat":
        sourcespace = "func1pt8"
        targetspace = "anat0pt8"
        voxelsize = 0.8
        res = 320
        # internal code used by NSD code to identify the correct transform file
        casenum = 1
        # specify transform file for functional to anatomical transformation
        transform_file = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_func1pt8-to-anat0pt8.nii.gz"
        )
        ext = ".nii.gz"
    elif transform_type == "anat_to_mni":
        sourcespace = "anat0pt8"
        targetspace = "mni"
        voxelsize = 1.0
        res = None
        # internal code used by NSD code to identify the correct transform file
        casenum = 1
        # specify transform file for anatomical to MNI transformation
        transform_file = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_anat0pt8-to-mni.nii.gz"
        )
        ext = ".nii.gz"
    elif transform_type == "func_to_mni":
        sourcespace = "func1pt8"
        targetspace = "mni"
        voxelsize = 1.0
        res = None
        # internal code used by NSD code to identify the correct transform file
        casenum = 1
        # specify transform file for functional to MNI transformation
        transform_file = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_func1pt8-to-mni.nii.gz"
        )
        ext = ".nii.gz"
    elif transform_type == "func_to_surface":
        sourcespace = "func1pt8"
        targetspace = "layerB3"
        voxelsize = None
        res = None
        # internal code used by NSD code to identify the correct transform file
        casenum = 2
        # specify transform file for functional to surface transformation
        transform_file = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_lh_func1pt8-to-layerB3.mgz"
        )
        ext = ".mgz"

    # load source data
    sourcefp = functional_data_path if "func" in sourcespace else anatomical_data_path
    assert sourcefp is not None, "Source data path must be provided."
    source_img = nib.nifti1.load(sourcefp)
    source_data = source_img.get_fdata()
    source_class = source_data.dtype

    # set out file name
    if output_prefix is not None:
        output_filename = os.path.join(output_dir, f"{output_prefix}{ext}")
    else:
        # get filename without extension and add suffix indicating transformation type
        input_filename = os.path.basename(sourcefp).split(".")[0]
        output_filename = os.path.join(
            output_dir, f"{input_filename}_{transform_type}{ext}"
        )

    # load transform file
    if casenum == 1:
        transform_img = nib.nifti1.load(transform_file)
        transform_array = transform_img.get_fdata()  # X x Y x Z x 3
    else:
        # V x 3 (decimal coordinates) or V x 1 (index)
        transform_img = nib.load(transform_file)  # type: ignore
        transform_array = transform_img.get_fdata()  # type: ignore
        # get rid of extra dim - nsdcode/load_data.py
        transform_array = transform_array.reshape(
            [transform_array.shape[0], -1], order="F"
        )

    # define transformation arguments as expected by transform_data function in NSD code
    # collect arguments for transform_data
    transform_args = {
        "casenum": casenum,
        "sourcespace": sourcespace,
        "targetspace": targetspace,
        "interptype": interptype,
        "badval": None,
        "outputfile": output_filename,
        "outputclass": source_class,
        "voxelsize": voxelsize,
        "res": res,
        "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
    }

    # apply transform
    data = transform_data(
        a1_data=transform_array,
        sourcedata=source_data,
        tr_args=transform_args,
    )
    # import nibabel.freesurfer.mghformat as fsmgh

    # mgh0 = f"{NSD_DATA_DIR}/anat/subj{subject}/surf/lh.w-g.pct.mgh"
    # img = fsmgh.load(mgh0)
    # header = img.header
    # affine = img.affine

    # v_img = fsmgh.MGHImage(data, affine, header=header, extra={})

    # v_img.to_filename(output_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Transform functional data between coordinate systems using NSD code."
    )
    argparser.add_argument(
        "-s",
        "--subject",
        required=True,
        type=str,
        help="Subject to perform transformation for. Provide without subject prefix (e.g. '01' for subj01).",
    )
    argparser.add_argument(
        "-t",
        "--transform_type",
        required=True,
        type=str,
        choices=[
            "anat_to_func",
            "func_to_anat",
            "anat_to_mni",
            "func_to_mni",
            "func_to_surface",
        ],
        help="Type of transformation to perform.",
    )
    argparser.add_argument(
        "-f",
        "--func",
        required=False,
        type=str,
        help="Path to the functional data in subject space to be transformed.",
    )
    argparser.add_argument(
        "-a",
        "--anat",
        required=False,
        type=str,
        help="Path to the anatomical data in subject space to be transformed.",
    )
    argparser.add_argument(
        "-i",
        "--interp_type",
        default="cubic",
        type=str,
        choices=["nearest", "linear", "cubic"],
        help="Interpolation type to use for the transformation. Default is 'cubic'.",
    )
    argparser.add_argument(
        "-o",
        "--out_dir",
        # default is working directory
        default=".",
        required=False,
        type=str,
        help="Directory to save the transformed functional data.",
    )
    argparser.add_argument(
        "-p",
        "--output_prefix",
        default=None,
        required=False,
        type=str,
        help="Prefix for the output file name. If not provided,"
        "the default is will be the filename of the input functional or "
        "anatomical data with a suffix indicating the transformation type.",
    )

    args = argparser.parse_args()
    main(
        subject=args.subject,
        functional_data_path=args.func,
        anatomical_data_path=args.anat,
        transform_type=args.transform_type,
        output_dir=args.out_dir,
        output_prefix=args.output_prefix,
        interp_type=args.interp_type,
    )
