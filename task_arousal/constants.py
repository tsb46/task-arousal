"""
Constants for the task arousal analysis.
"""

import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# path to data directory
DATA_DIRECTORY_EUSKALIBUR = os.getenv("DATA_DIRECTORY_EUSKALIBUR", "data/euskalibur")
DATA_DIRECTORY_PAN = os.getenv("DATA_DIRECTORY_PAN", "data/pan")
# flag indicating if the data should be searched in the 'derivatives' directory
IS_DERIVED = os.getenv("IS_DERIVED", "false").lower() == "true"

# Brain masks
MASK_EUSKALIBUR = "templates/MNI152_T1_3mm_brain_mask_dil_euskalibur.nii.gz"
MASK_GM_EUSKALIBUR = "templates/MNI152_T1_3mm_gm_mask_euskalibur.nii.gz"
MASK_PAN = "templates/MNI152_T1_3mm_brain_mask_dil_pan.nii.gz"
MASK_GM_PAN = "templates/MNI152_T1_3mm_gm_mask_pan.nii.gz"
# Surface templates
SURFACE_LH = "templates/fsLR_den-32k_hemi-L_inflated.surf.gii"
SURFACE_RH = "templates/fsLR_den-32k_hemi-R_inflated.surf.gii"

# TR (Repetition Time) in seconds
TR_EUSKALIBUR = 1.5
TR_PAN = 1.355


# expected columns in event dataframe
EVENT_COLUMNS = ["onset", "duration", "trial_type"]


## Preprocessing parameters
# Slice timing reference, between 0 and 1 (middle slice)
SLICE_TIMING_REF = 0.5
# Number of dummy volumes to drop
DUMMY_VOLUMES = 10
# High-pass filter cutoff frequency for fmri
HIGHPASS = 0.01
# Full width at half maximum for Gaussian smoothing
FWHM_EUSKALIBUR = 4  # in mm
FWHM_PAN = 4  # in mm
# physio fields to extract from raw data
PHYSIO_COLUMNS_EUSKALIBUR = [
    "respiratory_effort",
    "cardiac",
    "respiratory_CO2",
    "respiratory_O2",
]
# physiological resample frequency (in Hz)
PHYSIO_RESAMPLE_F = 50
