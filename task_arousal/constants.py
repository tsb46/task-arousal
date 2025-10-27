"""
Constants for the task arousal analysis.
"""
import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# path to data directory
DATA_DIRECTORY_EUSKALIBUR = os.getenv('DATA_DIRECTORY_EUSKALIBUR', 'data/euskalibur')
DATA_DIRECTORY_HCP = os.getenv('DATA_DIRECTORY_HCP', 'data/hcp')
# flag indicating if the data should be searched in the 'derivatives' directory
IS_DERIVED = os.getenv('IS_DERIVED', 'false').lower() == 'true'

# Brain mask
MASK_EUSKALIBUR = 'templates/MNI152_T1_3mm_brain_mask_dil_euskalibur.nii.gz'
MASK_HCP = 'templates/MNI152_T1_2mm_brain_mask_hcp.nii.gz'
# TR (Repetition Time) in seconds
TR_EUSKALIBUR = 1.5
TR_HCP = 0.72
# Slice timing reference, between 0 and 1 (middle slice)
SLICE_TIMING_REF = 0.5

# expected columns in event dataframe
EVENT_COLUMNS = ['onset', 'duration', 'trial_type']