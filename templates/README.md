# Templates

Brain masks for each dataset

- **MNI152_T1_3mm_brain_mask_dil_euskalibur.nii.gz** : a 3mm MNI brain mask for the Euskalibur dataset. Note, the brain mask was dilated to pick up voxels in large draining veins and dura outside of brain tissue.
- **MNI152NLin2009cAsym_res-02_desc-fitlins_brain_mask_90coverage.nii.gz** - a 2mm MNI brain mask generated from overlap from the OpenNeuro fitlins contrast maps. Specifically, all voxels that are present across at least 90% of all contrast maps (see task_arousal/io/openneuro_fitlin_contrasts.json) are included in the mask.
