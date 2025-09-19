"""
Constants for the task arousal analysis.
"""

# path to data directory
DATA_DIRECTORY = "data"
# Brain mask
MASK = 'templates/MNI152_T1_2mm_brain_mask_dil.nii.gz'
# TR (Repetition Time) in seconds
TR = 1.5
# Slice timing reference, between 0 and 1 (middle slice)
SLICE_TIMING_REF = 0.5

# expected columns in event dataframe
EVENT_COLUMNS = ['onset', 'duration', 'trial_type']