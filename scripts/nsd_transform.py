"""
CLI utility that is a wrapper around the transform_data module to transform between
functional, anatomical, MNI and fsaverage spaces provided by the authors:

https://github.com/cvnlab/nsdcode

This only performs a subset of the tranformations provided by the NSD code, specifically:
(1) subject-volume to subject-anatomical space (1.8mm to 1mm)
(2) subject-anatomical to subject-volume space (0.8 mm to 1.8mm)
(3) subject-volume to MNI (FSL) space (1.8mm to MNI)
(4) subject-anatomical to MNI (FSL) space (0.8mm to MNI)
(5) subject-volume to subject native surface space (1.8mm to surface)
(6) subject-volume to fsaverage surface space (1.8mm to fsaverage)

"""

import argparse
import os
from pathlib import Path
from typing import Literal, TypedDict

import nibabel as nib
import numpy as np
from numpy import dtype as np_dtype

try:
    from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
except ImportError:
    GiftiDataArray = None  # type: ignore[assignment]
    GiftiImage = None  # type: ignore[assignment]

from nsdcode.transform_data import transform_data

# hard code NSD data directory
NSD_DATA_DIR = "data/nsd"
NSD_LAYERS = ["layerB1", "layerB2", "layerB3"]


def _surface_gifti_suffix(data: np.ndarray) -> str:
    # take the data array, squeeze it to remove any singleton dimensions, and check the number of dimensions to determine
    # if it's functional (4D) or anatomical (3D) surface data, and return the appropriate suffix for the output gifti file name
    return ".func.gii" if np.asarray(data).squeeze().ndim > 3 else ".shape.gii"


def _save_surface_gifti(path: str, data: np.ndarray) -> None:
    if GiftiImage is None or GiftiDataArray is None:
        raise RuntimeError("nibabel.gifti is not available; cannot write GIFTI")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    data_arr = np.asarray(data, dtype=np.float32).squeeze()
    if data_arr.ndim == 0:
        raise ValueError("Surface data must contain at least one vertex.")

    if data_arr.ndim == 1:
        darrays = [GiftiDataArray(data=data_arr, intent="NIFTI_INTENT_SHAPE")]
    else:
        data_2d = data_arr.reshape(data_arr.shape[0], -1)
        darrays = [
            GiftiDataArray(data=data_2d[:, frame], intent="NIFTI_INTENT_TIME_SERIES")
            for frame in range(data_2d.shape[1])
        ]

    nib.save(GiftiImage(darrays=darrays), str(path_obj))  # type: ignore[arg-type]


# create a typed dict for the transformation inputs expected by the NSD code
class TransformInputs(TypedDict):
    sourcespace: str | None
    targetspace: str | None
    interp_type: Literal["nearest", "linear", "cubic"] | None
    sourcedata: np.ndarray | None
    sourceclass: np_dtype | str | None
    transformdata: np.ndarray | None
    casenum: int | None
    voxelsize: float | None
    res: int | None
    output_file_name: str | None
    output_file_name_lh: str | None  # only used for surface transformations
    output_file_name_rh: str | None  # only used for surface transformations
    transform_file: (
        str | None
    )  # only used for volume to volume transformations, not surface transformations
    transform_file_lh: str | None  # only used for functional to surface transformation
    transform_file_rh: str | None  # only used for functional to surface transformation
    transform_data_lh: (
        np.ndarray | None
    )  # only used for functional to surface transformation
    transform_data_rh: (
        np.ndarray | None
    )  # only used for functional to surface transformation
    transform_file_native_lh: (
        list[str] | None
    )  # only used for functional to fsaverage transformation
    transform_file_native_rh: (
        list[str] | None
    )  # only used for functional to fsaverage transformation
    transform_data_native_lh: (
        list[np.ndarray] | None
    )  # only used for functional to fsaverage transformation
    transform_data_native_rh: (
        list[np.ndarray] | None
    )  # only used for functional to fsaverage transformation
    transform_fsaverage_lh: (
        str | None
    )  # only used for functional to fsaverage transformation
    transform_fsaverage_rh: (
        str | None
    )  # only used for functional to fsaverage transformation
    transform_data_fsaverage_lh: (
        np.ndarray | None
    )  # only used for functional to fsaverage transformation
    transform_data_fsaverage_rh: (
        np.ndarray | None
    )  # only used for functional to fsaverage transformation
    ext: (
        Literal[".nii.gz", ".gii"] | None
    )  # file extension for output file, e.g. .nii.gz or .mgz


def main(
    subject: str,
    functional_data_path: str | None,
    anatomical_data_path: str | None,
    transform_type: Literal[
        "anat_to_func",
        "func_to_anat",
        "func_to_surface",
        "func_to_mni",
        "anat_to_mni",
        "func_to_fsaverage",
    ],
    output_dir: str,
    output_prefix: str | None = None,
    interp_type: Literal["nearest", "linear", "cubic"] = "cubic",
):
    """
    Main function to perform transformation between coordinate systems using NSD code.
    Notes:
        * surface files are always written to gifti format for visualization outside freesurfer.
        * the func_to_fsaverage transformation involves two transforms: 1) volume to all surface depths (layer 1,
            layer 2, layer 3) of the native surface, 2) average across depths and 3) nativesurface to fsaverage.
        * the func_to_surface transformation only transforms to layer b3.

    Parameters
    ----------
    subject: str
        Subject to perform transformation for. Provide without subject prefix (e.g. '01' for subj01).
    functional_data_path: str or None
        Path to the functional data in subject native space to be transformed. If not provided, the functional data will be loaded from the NSD data directory.
    anatomical_data_path: str or None
        Path to the anatomical data in subject anatomical space to be transformed. If not provided, the anatomical data will be loaded from the NSD data directory.
    transform_type: str
        Type of transformation to perform. One of "anat_to_func", "func_to_anat", "anat_to_mni", "func_to_mni", "func_to_surface" or "func_to_fsaverage".
    output_dir: str
        Directory to save the transformed data.
    output_prefix: str or None
        Prefix for the output file name. If not provided, the default will be the filename of the input functional or anatomical data with a suffix indicating the transformation type.
    """

    # must provide anatomical or functional data path depending on the transform type
    if transform_type == "anat_to_func" or transform_type == "anat_to_mni":
        if anatomical_data_path is None:
            raise ValueError(
                "Anatomical data path must be provided for anat_to_func or anat_to_mni transformation."
            )
    elif (
        transform_type == "func_to_anat"
        or transform_type == "func_to_mni"
        or transform_type == "func_to_surface"
        or transform_type == "func_to_fsaverage"
    ):
        if functional_data_path is None:
            raise ValueError(
                "Functional data path must be provided for func_to_anat, func_to_mni, func_to_surface, or func_to_fsaverage transformation."
            )

    # initialize variables for transform inputs
    transform_inputs: TransformInputs = {
        "sourcespace": None,
        "targetspace": None,
        "interp_type": interp_type,
        "sourcedata": None,
        "sourceclass": None,
        "transformdata": None,
        "voxelsize": None,
        "res": None,
        "output_file_name": None,
        "output_file_name_lh": None,
        "output_file_name_rh": None,
        "transform_file": None,
        "transform_file_lh": None,
        "transform_file_rh": None,
        "transform_data_lh": None,
        "transform_data_rh": None,
        "transform_file_native_lh": None,
        "transform_file_native_rh": None,
        "transform_data_native_lh": None,
        "transform_data_native_rh": None,
        "transform_fsaverage_lh": None,
        "transform_fsaverage_rh": None,
        "transform_data_fsaverage_lh": None,
        "transform_data_fsaverage_rh": None,
        "casenum": None,
        "ext": None,
    }

    # create transform inputs based on transform type
    if transform_type == "anat_to_func":
        transform_inputs["sourcespace"] = "anat0pt8"
        transform_inputs["targetspace"] = "func1pt8"
        transform_inputs["voxelsize"] = 1.8
        transform_inputs["res"] = None
        # internal code used by NSD code to identify the correct transform file
        transform_inputs["casenum"] = 1
        # specify transform file for anatomical to functional transformation
        transform_inputs["transform_file"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_anat0pt8-to-func1pt8.nii.gz"
        )
        transform_inputs["ext"] = ".nii.gz"
    elif transform_type == "func_to_anat":
        transform_inputs["sourcespace"] = "func1pt8"
        transform_inputs["targetspace"] = "anat0pt8"
        transform_inputs["voxelsize"] = 0.8
        transform_inputs["res"] = 320
        # internal code used by NSD code to identify the correct transform file
        transform_inputs["casenum"] = 1
        # specify transform file for functional to anatomical transformation
        transform_inputs["transform_file"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_func1pt8-to-anat0pt8.nii.gz"
        )
        transform_inputs["ext"] = ".nii.gz"
    elif transform_type == "anat_to_mni":
        transform_inputs["sourcespace"] = "anat0pt8"
        transform_inputs["targetspace"] = "mni"
        transform_inputs["voxelsize"] = 1.0
        transform_inputs["res"] = None
        # internal code used by NSD code to identify the correct transform file
        transform_inputs["casenum"] = 1
        # specify transform file for anatomical to MNI transformation
        transform_inputs["transform_file"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_anat0pt8-to-mni.nii.gz"
        )
        transform_inputs["ext"] = ".nii.gz"
    elif transform_type == "func_to_mni":
        transform_inputs["sourcespace"] = "func1pt8"
        transform_inputs["targetspace"] = "mni"
        transform_inputs["voxelsize"] = 1.0
        transform_inputs["res"] = None
        # internal code used by NSD code to identify the correct transform file
        transform_inputs["casenum"] = 1
        # specify transform file for functional to MNI transformation
        transform_inputs["transform_file"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_func1pt8-to-mni.nii.gz"
        )
        transform_inputs["ext"] = ".nii.gz"
    elif transform_type == "func_to_surface":
        transform_inputs["sourcespace"] = "func1pt8"
        transform_inputs["targetspace"] = "layerB3"
        transform_inputs["voxelsize"] = None
        transform_inputs["res"] = None
        # internal code used by NSD code to identify the correct transform file
        transform_inputs["casenum"] = 2
        # specify transform file for functional to surface transformation
        transform_inputs["transform_file_lh"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_lh_func1pt8-to-layerB3.mgz"
        )
        transform_inputs["transform_file_rh"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_rh_func1pt8-to-layerB3.mgz"
        )
        transform_inputs["ext"] = ".gii"
    elif transform_type == "func_to_fsaverage":
        transform_inputs["sourcespace"] = "func1pt8"
        transform_inputs["targetspace"] = "fsaverage"
        transform_inputs["voxelsize"] = None
        transform_inputs["res"] = None
        # internal code used by NSD code to identify the correct transform file
        transform_inputs["casenum"] = 3
        # specify transform file for functional to fsaverage transformation
        transform_inputs["transform_file"] = (
            None  # not used for func to fsaverage transformation
        )
        transform_inputs["transform_file_native_lh"] = [
            f"{NSD_DATA_DIR}/transform/subj{subject}_lh_func1pt8-to-layerB1.mgz",
            f"{NSD_DATA_DIR}/transform/subj{subject}_lh_func1pt8-to-layerB2.mgz",
            f"{NSD_DATA_DIR}/transform/subj{subject}_lh_func1pt8-to-layerB3.mgz",
        ]
        transform_inputs["transform_file_native_rh"] = [
            f"{NSD_DATA_DIR}/transform/subj{subject}_rh_func1pt8-to-layerB1.mgz",
            f"{NSD_DATA_DIR}/transform/subj{subject}_rh_func1pt8-to-layerB2.mgz",
            f"{NSD_DATA_DIR}/transform/subj{subject}_rh_func1pt8-to-layerB3.mgz",
        ]
        transform_inputs["transform_fsaverage_lh"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_lh_white-to-fsaverage.mgz"
        )
        transform_inputs["transform_fsaverage_rh"] = (
            f"{NSD_DATA_DIR}/transform/subj{subject}_rh_white-to-fsaverage.mgz"
        )
        transform_inputs["ext"] = ".gii"
    else:
        raise ValueError(
            f"Invalid transform type: {transform_type}. Must be one of 'anat_to_func', 'func_to_anat', 'anat_to_mni', 'func_to_mni', 'func_to_surface' or 'func_to_fsaverage'."
        )

    # load source data
    sourcefp = (
        functional_data_path
        if "func" in transform_inputs["sourcespace"]
        else anatomical_data_path
    )
    assert sourcefp is not None, "Source data path must be provided."
    source_img = nib.nifti1.load(sourcefp)
    transform_inputs["sourcedata"] = source_img.get_fdata()
    transform_inputs["sourceclass"] = source_img.get_data_dtype()

    # set out file name
    if output_prefix is None:
        output_prefix = os.path.basename(sourcefp).split(".")[0]
        # add suffix to output prefix based on transform type
        output_prefix += f"_{transform_type}"

    if transform_type in ["func_to_surface", "func_to_fsaverage"]:
        output_filename_lh = os.path.join(
            output_dir,
            f"{output_prefix}_lh{_surface_gifti_suffix(transform_inputs['sourcedata'])}",
        )
        output_filename_rh = os.path.join(
            output_dir,
            f"{output_prefix}_rh{_surface_gifti_suffix(transform_inputs['sourcedata'])}",
        )
        transform_inputs["output_file_name_lh"] = output_filename_lh
        transform_inputs["output_file_name_rh"] = output_filename_rh
    else:
        output_filename = os.path.join(
            output_dir, f"{output_prefix}{transform_inputs['ext']}"
        )
        transform_inputs["output_file_name"] = output_filename

    # load transform file
    if transform_type in ["anat_to_func", "anat_to_mni", "func_to_anat", "func_to_mni"]:
        # check that transform file exists
        assert transform_inputs["transform_file"] is not None, (
            "Transform file must be specified for volume to volume transformations."
        )
        transform_img = nib.nifti1.load(transform_inputs["transform_file"])
        transform_inputs["transformdata"] = transform_img.get_fdata()  # X x Y x Z x 3
    else:
        # utility function for unpacking surface transform files provided by NSD code
        def _load_surface_transform(transform_file):
            transform_img = nib.load(transform_file)  # type: ignore
            transform_array = transform_img.get_fdata()  # type: ignore
            # get rid of extra dim - nsdcode/load_data.py
            transform_array = transform_array.reshape(
                [transform_array.shape[0], -1], order="F"
            )
            return transform_array

        if transform_type == "func_to_surface":
            # check that transform files for both hemispheres exist
            assert (
                transform_inputs["transform_file_lh"] is not None
                and transform_inputs["transform_file_rh"] is not None
            ), (
                "Transform files for both hemispheres must be specified for functional to surface transformation."
            )
            # for functional to surface transformation, we have separate transform files for left and right hemispheres
            transform_inputs["transform_data_lh"] = _load_surface_transform(
                transform_inputs["transform_file_lh"]
            )  # V x 3 (decimal coordinates) or V x 1 (index)
            transform_inputs["transform_data_rh"] = _load_surface_transform(
                transform_inputs["transform_file_rh"]
            )  # V x 3 (decimal coordinates) or V x 1 (index)
        elif transform_type == "func_to_fsaverage":
            # check that transform files for both hemispheres and all surface depths exist
            assert (
                transform_inputs["transform_file_native_lh"] is not None
                and transform_inputs["transform_file_native_rh"] is not None
                and transform_inputs["transform_fsaverage_lh"] is not None
                and transform_inputs["transform_fsaverage_rh"] is not None
            ), (
                "Transform files for both hemispheres and all surface depths must be specified for functional to fsaverage transformation."
            )
            # for functional to fsaverage transformation, we have separate transform files for left and right hemispheres and for each surface depth (layerB1, layerB2, layerB3)
            transform_inputs["transform_data_native_lh"] = [
                _load_surface_transform(transform_file)
                for transform_file in transform_inputs["transform_file_native_lh"]
            ]  # list of 3 arrays, each V x 3 (decimal coordinates) or V x 1 (index)
            transform_inputs["transform_data_native_rh"] = [
                _load_surface_transform(transform_file)
                for transform_file in transform_inputs["transform_file_native_rh"]
            ]  # list of 3 arrays, each V x 3 (decimal coordinates) or V x 1 (index)
            transform_inputs["transform_data_fsaverage_lh"] = _load_surface_transform(
                transform_inputs["transform_fsaverage_lh"]
            )  # V_fsaverage x 3 (decimal coordinates) or V_fsaverage x 1 (index)
            transform_inputs["transform_data_fsaverage_rh"] = _load_surface_transform(
                transform_inputs["transform_fsaverage_rh"]
            )  # V_fsaverage x 3 (decimal coordinates) or V_fsaverage x 1 (index)

    # define transformation arguments as expected by transform_data function in NSD code
    # perform volume-to-volume transformation
    if transform_type in ["anat_to_func", "anat_to_mni", "func_to_anat", "func_to_mni"]:
        # collect arguments for transform_data
        transform_args = {
            "casenum": transform_inputs["casenum"],
            "sourcespace": transform_inputs["sourcespace"],
            "targetspace": transform_inputs["targetspace"],
            "interptype": transform_inputs["interp_type"],
            "badval": None,
            "outputfile": transform_inputs["output_file_name"],
            "outputclass": transform_inputs["sourceclass"],
            "voxelsize": transform_inputs["voxelsize"],
            "res": transform_inputs["res"],
            "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
        }

        # apply transform
        transform_data(
            a1_data=transform_inputs["transformdata"],
            sourcedata=transform_inputs["sourcedata"],
            tr_args=transform_args,
        )
    # perform functional to surface transformation
    elif transform_type == "func_to_surface":
        # collect arguments for transform_data
        transform_args_lh = {
            "casenum": transform_inputs["casenum"],
            "sourcespace": transform_inputs["sourcespace"],
            "targetspace": transform_inputs["targetspace"],
            "interptype": transform_inputs["interp_type"],
            "badval": None,
            "outputfile": None,
            "outputclass": transform_inputs["sourceclass"],
            "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
        }
        transform_args_rh = {
            "casenum": transform_inputs["casenum"],
            "sourcespace": transform_inputs["sourcespace"],
            "targetspace": transform_inputs["targetspace"],
            "interptype": transform_inputs["interp_type"],
            "badval": None,
            "outputfile": None,
            "outputclass": transform_inputs["sourceclass"],
            "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
        }

        # transform left hemisphere
        data_lh = transform_data(
            a1_data=transform_inputs["transform_data_lh"],
            sourcedata=transform_inputs["sourcedata"],
            tr_args=transform_args_lh,
        )
        # transform right hemisphere
        data_rh = transform_data(
            a1_data=transform_inputs["transform_data_rh"],
            sourcedata=transform_inputs["sourcedata"],
            tr_args=transform_args_rh,
        )
        assert (
            transform_inputs["output_file_name_lh"] is not None
            and transform_inputs["output_file_name_rh"] is not None
        ), (
            "Output file names for both hemispheres must be specified for functional to surface transformation."
        )
        _save_surface_gifti(transform_inputs["output_file_name_lh"], data_lh)
        _save_surface_gifti(transform_inputs["output_file_name_rh"], data_rh)
    # perform functional to fsaverage transformation
    elif transform_type == "func_to_fsaverage":
        native_depth_lh = []
        native_depth_rh = []
        # first, loop through depths (layerB1, layerB2, layerB3) and transform to native surface for each hemisphere
        for depth in NSD_LAYERS:
            # collect arguments for transform_data for left hemisphere
            transform_args_lh = {
                "casenum": 2,
                "sourcespace": transform_inputs["sourcespace"],
                "targetspace": depth,
                "interptype": transform_inputs["interp_type"],
                "badval": None,
                "outputfile": None,  # not saving intermediate native surface files
                "outputclass": transform_inputs["sourceclass"],
                "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
            }
            # collect arguments for transform_data for right hemisphere
            transform_args_rh = {
                "casenum": 2,
                "sourcespace": transform_inputs["sourcespace"],
                "targetspace": depth,
                "interptype": transform_inputs["interp_type"],
                "badval": None,
                "outputfile": None,  # not saving intermediate native surface files
                "outputclass": transform_inputs["sourceclass"],
                "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
            }

            # transform left hemisphere to native surface for current depth
            assert transform_inputs["transform_data_native_lh"] is not None, (
                "Transform data for left hemisphere native surface transformation must be provided."
            )
            data_lh_t = transform_data(
                a1_data=transform_inputs["transform_data_native_lh"][
                    NSD_LAYERS.index(depth)
                ],
                sourcedata=transform_inputs["sourcedata"],
                tr_args=transform_args_lh,
            )
            native_depth_lh.append(data_lh_t)
            # transform right hemisphere to native surface for current depth
            assert transform_inputs["transform_data_native_rh"] is not None, (
                "Transform data for right hemisphere native surface transformation must be provided."
            )
            data_rh_t = transform_data(
                a1_data=transform_inputs["transform_data_native_rh"][
                    NSD_LAYERS.index(depth)
                ],
                sourcedata=transform_inputs["sourcedata"],
                tr_args=transform_args_rh,
            )
            native_depth_rh.append(data_rh_t)
        # average across depths for each hemisphere
        native_avg_lh = np.mean(native_depth_lh, axis=0)
        native_avg_rh = np.mean(native_depth_rh, axis=0)
        # then, transform from native surface to fsaverage for each hemisphere
        # collect arguments for transform_data for left hemisphere
        transform_args_lh = {
            "casenum": transform_inputs["casenum"],
            "sourcespace": transform_inputs["targetspace"],
            "targetspace": "fsaverage",
            "interptype": "nearest",  # for surface to surface transformation, use nearest neighbor interpolation
            "badval": None,
            "outputfile": None,
            "outputclass": transform_inputs["sourceclass"],
            "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
        }
        # collect arguments for transform_data for right hemisphere
        transform_args_rh = {
            "casenum": transform_inputs["casenum"],
            "sourcespace": transform_inputs["targetspace"],
            "targetspace": "fsaverage",
            "interptype": "nearest",  # for surface to surface transformation, use nearest neighbor interpolation
            "badval": None,
            "outputfile": None,
            "outputclass": transform_inputs["sourceclass"],
            "fsdir": f"{NSD_DATA_DIR}/anat/subj{subject}",
        }

        # transform left hemisphere from native surface to fsaverage
        assert transform_inputs["transform_data_fsaverage_lh"] is not None, (
            "Transform data for left hemisphere fsaverage transformation must be provided."
        )
        fsaverage_lh = transform_data(
            a1_data=transform_inputs["transform_data_fsaverage_lh"],
            sourcedata=native_avg_lh,
            tr_args=transform_args_lh,
        )
        # transform right hemisphere from native surface to fsaverage
        assert transform_inputs["transform_data_fsaverage_rh"] is not None, (
            "Transform data for right hemisphere fsaverage transformation must be provided."
        )
        fsaverage_rh = transform_data(
            a1_data=transform_inputs["transform_data_fsaverage_rh"],
            sourcedata=native_avg_rh,
            tr_args=transform_args_rh,
        )

        assert (
            transform_inputs["output_file_name_lh"] is not None
            and transform_inputs["output_file_name_rh"] is not None
        ), (
            "Output file names for both hemispheres must be specified for functional to fsaverage transformation."
        )
        _save_surface_gifti(transform_inputs["output_file_name_lh"], fsaverage_lh)
        _save_surface_gifti(transform_inputs["output_file_name_rh"], fsaverage_rh)


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
            "func_to_fsaverage",
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
