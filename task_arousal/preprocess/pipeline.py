"""
Preprocessing pipeline for fMRI and physiological data.

fMRI volume or surface preprocessing is performed on outputs from fMRIPrep for EuskalIBUR, and Precision
Association Networks, including:

1) Drop dummy volumes
2) Detrending
3) High-pass filtering (> 0.01 Hz)
4) Standardize signal (z-score)
5) Smoothing

Physio preprocessing is performed on raw physiological data, including:

1) Feature extraction
2) High-pass filtering
3) Resampling to fMRI time points
"""

import os
import warnings

from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

from task_arousal.constants import (
    MASK_EUSKALIBUR,  # brain mask for EuskalIBUR
    MASK_PAN,  #  brain mask for PAN
    TR_EUSKALIBUR,  # TR for EuskalIBUR
    TR_PAN,  # TR for PAN
    SLICE_TIMING_REF,  # slice timing reference
)

## Preprocessing parameters
from task_arousal.constants import (
    DUMMY_VOLUMES,  # number of dummy volumes to drop
    HIGHPASS,  # high-pass filter cutoff frequency
    FWHM_EUSKALIBUR,  # smoothing FWHM for EuskalIBUR
    FWHM_PAN,  # smoothing FWHM for PAN
    PHYSIO_COLUMNS_EUSKALIBUR,  # physio columns to extract for EuskalIBUR
    PHYSIO_RESAMPLE_F,  # physio resample frequency
    SURFACE_LH,  # left hemisphere surface template
    SURFACE_RH,  # right hemisphere surface template
)

from task_arousal.io.file import FileMapper
from task_arousal.preprocess.components.physio import physio_pipeline
from task_arousal.preprocess.components.volume import func_volume_pipeline
from task_arousal.preprocess.components.surface import func_surface_pipeline


class PreprocessingPipeline:
    """
    Preprocessing pipeline for fMRI and physiological data.
    """

    def __init__(
        self,
        dataset: Literal["euskalibur", "pan"],
        subject: str,
        func_type: Literal["volume", "surface"] = "volume",
    ) -> None:
        """Initialize the preprocessing pipeline for a specific dataset and subject.

        Parameters
        ----------
            dataset (Literal['euskalibur', 'pan']): The dataset identifier.
            subject (str): The subject identifier.
            func_type (Literal['volume', 'surface'], optional): The type of functional data. Defaults to "volume".
        """
        self.subject = subject
        self.dataset = dataset
        self.func_type: Literal["volume", "surface"] = func_type

        # map file paths associated to subject
        if dataset in ["euskalibur", "pan"]:
            self.file_mapper = FileMapper(dataset, subject)
        else:
            raise ValueError(f"Dataset '{dataset}' is not supported.")
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks

    def preprocess(
        self,
        task: str | None = None,
        sessions: list[str] | None = None,
        skip_func: bool = False,
        skip_physio: bool = False,
        save_physio_figs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Preprocess fMRI and physiological data.

        Parameters
        ----------
        task : str or None, optional
            The task identifier. If no task identifier is provided, all tasks will be processed. Defaults to None.
        sessions : list of str or None, optional
            The session identifiers. If no session identifiers are provided, all sessions will be processed. Defaults to None.
        skip_func : bool, optional
            Whether to skip fMRI preprocessing. Defaults to False.
        skip_physio : bool, optional
            Whether to skip physiological preprocessing. Defaults to False.
        """
        # pan has no physio data
        if self.dataset == "pan":
            if skip_physio:
                warnings.warn(
                    "Skipping physiological preprocessing for PAN dataset has no effect as there is no physio data."
                )
            skip_physio = True
        # check that not both func and physio are skipped
        if skip_func and skip_physio:
            raise ValueError(
                "Both fMRI and physiological preprocessing cannot be skipped."
            )
        if skip_func and verbose:
            print(f"Skipping fMRI preprocessing for subject '{self.subject}'...")
        if skip_physio and verbose and not self.dataset == "pan":
            print(
                f"Skipping physiological preprocessing for subject '{self.subject}'..."
            )

        # if task is not None, ensure it's an available task
        if task is not None:
            if task not in self.tasks:
                raise ValueError(
                    f"Task '{task}' is not available for subject '{self.subject}'."
                )
            tasks_to_process = [task]
        else:
            tasks_to_process = self.tasks
        if verbose:
            print(f"Processing tasks for subject '{self.subject}': {tasks_to_process}")

        # loop through each task and process scans and physio from all sessions (and runs)
        for task_proc in tasks_to_process:
            if verbose:
                print(
                    f"Processing task '{task_proc}' for subject '{self.subject}' "
                    f"and sessions '{sessions if sessions is not None else 'all'}'..."
                )

            # functional MRI preprocessing
            if not skip_func:
                # get fmri files for task
                fmri_files = self.file_mapper.get_fmri_files(
                    task_proc, sessions=sessions, func_type=self.func_type
                )
                # loop through fmri files and preprocess
                for fmri_file in fmri_files:
                    if verbose:
                        print(f"Preprocessing fMRI file: {fmri_file}")

                    # get metadata based on dataset
                    if self.dataset == "euskalibur":
                        fwhm = FWHM_EUSKALIBUR
                        tr = TR_EUSKALIBUR
                        mask = MASK_EUSKALIBUR
                        resample = False
                        remove_dummy = True
                    elif self.dataset == "pan":
                        fwhm = FWHM_PAN
                        tr = TR_PAN
                        mask = MASK_PAN
                        resample = True
                        remove_dummy = True
                    else:
                        raise ValueError(f"Unknown dataset: {self.dataset}")
                    # Apply the functional MRI preprocessing pipeline
                    if self.func_type == "volume":
                        fmri_proc = func_volume_pipeline(
                            func_fp=fmri_file,
                            tr=tr,
                            brain_mask_fp=mask,
                            fwhm=fwhm,
                            dummy_vols=DUMMY_VOLUMES,
                            highpass=HIGHPASS,
                            resample=resample,
                            remove_dummy=remove_dummy,
                        )
                    elif self.func_type == "surface":
                        fmri_proc = func_surface_pipeline(
                            func_fp=fmri_file,
                            tr=tr,
                            dummy_vols=DUMMY_VOLUMES,
                            highpass=HIGHPASS,
                            fwhm=fwhm,
                            remove_dummy=remove_dummy,
                            surface_template_lh=SURFACE_LH,
                            surface_template_rh=SURFACE_RH,
                        )
                    else:
                        raise ValueError(f"Unknown functional type: {self.func_type}")
                    # Write out the preprocessed fMRI file
                    self.write_out_fmri_file(fmri_proc, fmri_file)

            # physio preprocessing pipeline
            if not skip_physio:
                # get physio files (and JSON sidecars) for task
                physio_files = self.file_mapper.get_physio_files(
                    task_proc, return_json=True, sessions=sessions
                )
                # loop through physio files and preprocess
                for physio_file in physio_files:
                    if verbose:
                        print(f"Preprocessing physiological file: {physio_file[0]}")

                    # find matching fmri file to get number of volumes for resampling
                    # different techniques based on dataset
                    file_entities = self.file_mapper.layout.parse_file_entities(
                        physio_file[0]
                    )
                    matching_fmri_files = self.file_mapper.get_matching_files(
                        file_entities, "fmri"
                    )

                    if len(matching_fmri_files) == 0:
                        # in some scenarios, physio may be recorded but no usable fMRI data
                        if verbose:
                            print(
                                f"No matching fMRI file found for physiological file '{physio_file[0]}'."
                            )
                        continue

                    elif len(matching_fmri_files) > 1:
                        raise ValueError(
                            f"Multiple matching fMRI files found for physiological file '{physio_file[0]}'. "
                        )
                    else:
                        matching_fmri_file = matching_fmri_files[0]
                        if verbose:
                            print(f"Found matching fMRI file: {matching_fmri_file}")

                    # Apply the physiological preprocessing pipeline
                    physio_proc = physio_pipeline(
                        dataset=self.dataset,
                        physio_fp=physio_file[0],
                        physio_json=physio_file[1],
                        fmri_fp=matching_fmri_file,
                        tr=tr,  # type: ignore
                        fmri_dummy_n=DUMMY_VOLUMES,
                        highpass=HIGHPASS,
                        physio_resample_f=PHYSIO_RESAMPLE_F,
                        slicetiming_ref=SLICE_TIMING_REF,
                        # only Euskalibur has physio columns defined
                        physio_cols=PHYSIO_COLUMNS_EUSKALIBUR,
                        save_physio_figs=save_physio_figs,
                    )
                    # Write out the preprocessed physiological file
                    self.write_out_physio_file(physio_proc, physio_file[0])

            if verbose:
                print(
                    f"Finished processing task '{task}' for subject '{self.subject}'."
                )

    def write_out_fmri_file(
        self,
        fmri_img: nib.nifti1.Nifti1Image | nib.cifti2.cifti2.Cifti2Image,
        file_orig: str,
    ) -> None:
        """Write out the preprocessed fMRI file.

        Parameters
        ----------
        fmri_img : nib.nifti1.Nifti1Image | nib.cifti2.cifti2.Cifti2Image
            The preprocessed fMRI image.
        file_orig : str
            The original file path
        """
        # get output directory from original file path
        output_dir = self.file_mapper.get_out_directory(file_orig)
        # get file name from original file path
        file_orig_name = os.path.basename(file_orig)

        def _strip_known_fmri_extensions(name: str) -> str:
            for ext in (".nii.gz", ".dtseries.nii", ".nii"):
                if name.endswith(ext):
                    return name[: -len(ext)]
            return name

        def _make_desc_preprocfinal_name(name: str, out_ext: str) -> str:
            """Create a BIDS-ish output filename with `desc-preprocfinal`."""
            base = _strip_known_fmri_extensions(name)

            if "desc-preprocfinal" in base:
                new_base = base
            elif "desc-preproc" in base:
                new_base = base.replace("desc-preproc", "desc-preprocfinal", 1)
            elif "_bold" in base:
                new_base = base.replace("_bold", "_desc-preprocfinal_bold", 1)
            else:
                new_base = f"{base}_desc-preprocfinal"

            return f"{new_base}{out_ext}"

        if self.func_type == "volume":
            # NIfTI outputs are always `.nii.gz` in this project.
            output_name = _make_desc_preprocfinal_name(file_orig_name, ".nii.gz")
            output_path = f"{output_dir}/{output_name}"

            if not isinstance(fmri_img, nib.nifti1.Nifti1Image):
                raise TypeError(
                    f"Expected NIfTI image for volume output, got {type(fmri_img)}"
                )
            nib.nifti1.save(fmri_img, output_path)

        elif self.func_type == "surface":
            output_name = _make_desc_preprocfinal_name(file_orig_name, ".dtseries.nii")
            output_path = f"{output_dir}/{output_name}"

            if not isinstance(fmri_img, nib.cifti2.cifti2.Cifti2Image):
                raise TypeError(
                    f"Expected CIFTI image for surface output, got {type(fmri_img)}"
                )
            fmri_img.to_filename(output_path)

        else:
            raise ValueError(f"Unknown functional type: {self.func_type}")

    def write_out_physio_file(
        self, physio_dict: dict[str, np.ndarray], file_orig: str
    ) -> None:
        """Write out the preprocessed physiological file.

        Parameters
        ----------
        physio_dict : dict[str, np.ndarray]
            The preprocessed physiological data.
        file_orig : str
            The original file path
        """
        # convert to dataframe
        physio_df = pd.DataFrame(physio_dict)
        # get output directory from original file path
        output_dir = self.file_mapper.get_out_directory(file_orig)
        # get file name from original file path
        file_orig_name = os.path.basename(file_orig)
        # strip 'physio.tsv.gz' from file name
        file_new_name = file_orig_name.rstrip("physio.tsv.gz")
        # add 'desc-preproc_physio.tsv.gz' to file name
        file_new = f"{file_new_name}desc-preproc_physio.tsv.gz"
        # write out as tsv.gz with pandas
        physio_df.to_csv(
            f"{output_dir}/{file_new}", sep="\t", index=False, compression="gzip"
        )
