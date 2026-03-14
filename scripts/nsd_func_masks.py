"""
The functional brain masks provided by NSD are not suitable for our analysis, as they extend far beyond
the brain and include many non-brain voxels. This script creates new masks for each subject that are more
appropriate for our analysis. We utilize the transformation files provided by NSD to transform the anatomical brain masks
from the anatomical space to the functional space. The steps are as follows:


1) BET (skull-stripping) of the anatomical images to create a brain mask for each subject.
2) Resampling the brain masks to match the functional data resolution.
"""

import os
import nibabel as nib

from nsdcode.transform_data import transform_data

DATA_DIRECTORY_NSD = "data/nsd"
ANAT_DIR = f"{DATA_DIRECTORY_NSD}/anat"
OUTPUT_DIR = f"{DATA_DIRECTORY_NSD}/mask"


def main(subject: str):
    """
    Main function to create functional brain masks for NSD subjects.

    Parameters
    ----------
    subject: str
        Subject to create brain mask for. Provide without subject prefix (e.g. '01' for subj01).
    output_dir: str
        Directory to save the transformed brain masks.
    """
    print(f"Creating functional brain mask for subject {subject}...")
    # define paths to anatomical data and output mask
    anatomical_data_path = f"{ANAT_DIR}/subj{subject}_T1w_0pt8.nii.gz"

    # first, create a brain mask in anatomical space using BET (skull-stripping)
    os.system(
        f"bet {anatomical_data_path} {OUTPUT_DIR}/subj{subject}_anat_brain -m -f 0.3"
    )
    # the above command creates a brain mask with the suffix '_anat_brain_mask.nii.gz' in the output directory
    # remove the masked anatomical image created by BET, we only need the brain mask
    os.remove(f"{OUTPUT_DIR}/subj{subject}_anat_brain.nii.gz")

    # dilate the brain mask by 2 voxels to ensure it covers the entire brain in functional space
    # use fslmaths command to dilate the brain mask
    brain_mask_path = f"{OUTPUT_DIR}/subj{subject}_anat_brain_mask.nii.gz"
    os.system(f"fslmaths {brain_mask_path} -dilF -dilF {brain_mask_path}")

    # second, perform a nearest neighbor transformation from anatomical space to functional space
    # get the transformation file from anatomical space to functional space provided by NSD
    transform_file = (
        f"{DATA_DIRECTORY_NSD}/transform/subj{subject}_anat0pt8-to-func1pt8.nii.gz"
    )
    transform_img = nib.nifti1.load(transform_file)
    transform_array = transform_img.get_fdata()
    # load the anatomical brain mask
    anat_mask_img = nib.load(brain_mask_path)
    anat_mask_data = anat_mask_img.get_fdata()
    anat_mask_data_class = anat_mask_data.dtype
    # define transformation arguments for anat_to_func transformation
    # collect arguments for transform_data
    transform_args = {
        "casenum": 1,
        "sourcespace": "anat0pt8",
        "targetspace": "func1pt8",
        "interptype": "nearest",
        "badval": None,
        "outputfile": f"{OUTPUT_DIR}/subj{subject}_func1pt8mm_brain_mask.nii.gz",
        "outputclass": anat_mask_data_class,
        "voxelsize": 1.8,
        "res": None,
        "fsdir": f"{DATA_DIRECTORY_NSD}/anat/subj{subject}",
    }
    # apply transform
    transform_data(
        a1_data=transform_array,
        sourcedata=anat_mask_data,
        tr_args=transform_args,
    )

    # delete the anatomical brain mask in anatomical space, we only need the transformed mask in functional space
    os.remove(brain_mask_path)


if __name__ == "__main__":
    # create functional brain masks for all subjects
    for subject in range(1, 8):
        main(str(subject).zfill(2))
