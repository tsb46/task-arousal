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

from typing import Literal, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from task_arousal.constants import (
    MASK_EUSKALIBUR,  # brain mask for EuskalIBUR
    MASK_PAN,  #  brain mask for PAN
    PHYSIO_COLUMNS_EUSKALIBUR,  # physio columns to extract for EuskalIBUR
    PHYSIO_COLUMNS_NSD,  # physio columns to extract for NSD
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
    FWHM_NSD,  # smoothing FWHM for NSD
    PHYSIO_RESAMPLE_F,  # physio resample frequency
    SURFACE_LH,  # left hemisphere surface template
    SURFACE_RH,  # right hemisphere surface template
)

from task_arousal.io.file import FileMapperBids, FileMapperNSD
from task_arousal.preprocess.components.physio import physio_pipeline
from task_arousal.preprocess.components.volume import func_volume_pipeline
from task_arousal.preprocess.components.surface import func_surface_pipeline

# Define a type for the file mapper, which can be either BIDS or NSD
FileMapper = FileMapperBids | FileMapperNSD


class PreprocessingPipeline:
    """
    Preprocessing pipeline for fMRI and physiological data.
    """

    def __init__(
        self,
        dataset: Literal["euskalibur", "pan", "nsd"],
        subject: str,
        func_type: Literal["volume", "surface"] = "volume",
    ) -> None:
        """Initialize the preprocessing pipeline for a specific dataset and subject.

        Parameters
        ----------
            dataset (Literal['euskalibur', 'pan', 'nsd']): The dataset identifier.
            subject (str): The subject identifier.
            func_type (Literal['volume', 'surface'], optional): The type of functional data. Defaults to "volume".
        """
        self.subject = subject
        self.dataset = dataset
        self.func_type: Literal["volume", "surface"] = func_type

        # map file paths associated to subject
        if dataset in ("euskalibur", "pan"):
            self.file_mapper = FileMapperBids(dataset=dataset, subject=subject)
        elif dataset == "nsd":
            self.file_mapper = FileMapperNSD(subject)
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
        if self.dataset in ["pan"]:
            if skip_physio:
                warnings.warn(
                    "Skipping physiological preprocessing for this dataset has no effect - no physio data."
                )
            skip_physio = True
        # nsd has no physio data for nsdimagery dataset, but has physio data for rest dataset, so we only skip physio if the task is nsdimagery
        elif self.dataset == "nsd":
            if task == "nsdimagery":
                if skip_physio:
                    warnings.warn(
                        "Skipping physiological preprocessing for this task in NSD dataset has no effect - no physio data for nsdimagery task."
                    )
                skip_physio = True
        # check that not both func and physio are skipped
        if skip_func and skip_physio:
            raise ValueError(
                "Both fMRI and physiological preprocessing cannot be skipped."
            )
        if skip_func and verbose:
            print(f"Skipping fMRI preprocessing for subject '{self.subject}'...")
        if skip_physio and verbose and self.dataset not in ["pan"]:
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

            # Resolve dataset-specific parameters once for this task.
            if self.dataset == "euskalibur":
                fwhm = FWHM_EUSKALIBUR
                tr = TR_EUSKALIBUR
                mask = MASK_EUSKALIBUR
                resample = False
                remove_dummy = True
                physio_columns = PHYSIO_COLUMNS_EUSKALIBUR
            elif self.dataset == "pan":
                fwhm = FWHM_PAN
                tr = TR_PAN
                mask = MASK_PAN
                resample = True
                remove_dummy = True
                physio_columns = []  # PAN dataset does not have physio data, so no columns to extract
            elif self.dataset == "nsd":
                if not isinstance(self.file_mapper, FileMapperNSD):
                    raise TypeError(
                        "NSD dataset requires FileMapperNSD, got "
                        f"{type(self.file_mapper)}"
                    )
                fwhm = FWHM_NSD
                # TR is handled in the file mapper class for NSD since it differs by task, we retrieve it from the file mapper here
                tr = self.file_mapper.get_tr(task_proc)
                # subject functional masks are different for each subject and are generated as part of the additional preprocessing steps, so
                # we retrieve the subject-specific mask here rather than using a constant template mask as in the other datasets
                mask = self.file_mapper.get_subject_mask()
                resample = False
                remove_dummy = False  # NSD data is already preprocessed and does not have dummy volumes, so we do not want to remove any volumes here
                physio_columns = PHYSIO_COLUMNS_NSD
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")

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

            # physio preprocessing pipeline (EuskalIBUR and NSD only)
            if self.dataset in ["euskalibur", "nsd"] and not skip_physio:
                # if NSD dataset and task is nsdimagery, skip physio preprocessing since there is no physio data for that task
                if self.dataset == "nsd" and task_proc == "nsdimagery":
                    if verbose:
                        print(
                            "Skipping physiological preprocessing for nsdimagery task in NSD dataset - no physio data available."
                        )
                    continue
                # get physio files (and JSON sidecars) for task
                physio_files = self.file_mapper.get_physio_files(
                    task_proc, return_json=True, sessions=sessions
                )
                # loop through physio files and preprocess
                for physio_file in physio_files:
                    if verbose:
                        print(f"Preprocessing physiological file(s): {physio_file[0]}")

                    # find matching fmri file to get number of volumes for resampling
                    # first, parse the file components to get the session and run identifiers, which are needed
                    # to find the matching fMRI file. The parsing is dataset-specific since file naming conventions
                    # differ across datasets, so we use different parsing techniques based on the dataset.
                    if self.dataset == "euskalibur":
                        # for type checking
                        assert isinstance(self.file_mapper, FileMapperBids)
                        file_entities = self.file_mapper.layout.parse_file_entities(
                            physio_file[0]
                        )
                    elif self.dataset == "nsd":
                        # for type checking
                        assert isinstance(self.file_mapper, FileMapperNSD)
                        # NSD physio files are organized by session, but fMRI files are not, so we need to
                        # find the matching fMRI file by matching the session identifier in the file name
                        # we pass just the pulse physio file to the parser since NSD physio files are missing the json sidecar,
                        # and we just need the session identifier which is in the file name
                        file_entities = (
                            self.file_mapper._parse_func_file_list_components(
                                file_list=[physio_file[0][0]]
                            )
                        )[0]
                    else:
                        raise ValueError(f"Unknown dataset: {self.dataset}")

                    # second, use the parsed file components to find the matching fMRI file. We expect exactly one matching fMRI file for each physio file, but in some cases there may be no matching fMRI file (e.g. if physio was recorded but all fMRI data was unusable and excluded from the dataset), so we handle both cases.
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

                    # type check to satisfy type checker that matching_fmri_file is a string and not a list
                    if not isinstance(matching_fmri_file, str):
                        raise TypeError(
                            f"Expected a single matching fMRI file path as a string, but got: {matching_fmri_file}"
                        )
                    # Apply the physiological preprocessing pipeline
                    physio_proc = physio_pipeline(
                        dataset=self.dataset,
                        physio_fp=physio_file[0],
                        physio_json=physio_file[1],
                        fmri_fp=matching_fmri_file,
                        tr=tr,
                        fmri_dummy_n=DUMMY_VOLUMES,
                        highpass=HIGHPASS,
                        remove_dummy=remove_dummy,
                        physio_resample_f=PHYSIO_RESAMPLE_F,
                        slicetiming_ref=SLICE_TIMING_REF,
                        physio_cols=physio_columns,
                        save_physio_figs=save_physio_figs,
                    )
                    # Write out the preprocessed physiological file
                    self.write_out_physio_file(physio_proc, physio_file[0])

            if verbose:
                print(
                    f"Finished processing task '{task_proc}' for subject '{self.subject}'."
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
        # for the Euskalbur and PAND datasets, preprocessed files are saved in the
        # same directory as original files. For NSD, preprocessed files are saved in a separate directory.
        if self.dataset in ["euskalibur", "pan"]:
            output_dir = self.file_mapper.get_out_directory(file_orig)
        elif self.dataset == "nsd":
            if not isinstance(self.file_mapper, FileMapperNSD):
                raise TypeError(
                    f"NSD dataset requires FileMapperNSD, got {type(self.file_mapper)}"
                )
            output_dir = self.file_mapper.data_directory + "/func/final"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

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
        self, physio_dict: dict[str, np.ndarray], file_orig: str | Tuple[str, str]
    ) -> None:
        """Write out the preprocessed physiological file.

        Parameters
        ----------
        physio_dict : dict[str, np.ndarray]
            The preprocessed physiological data.
        file_orig : str | Tuple[str, str]
             The original file path. For datasets with separate files for different physiological signals, this can be a tuple of file paths.
        """
        # convert to dataframe
        physio_df = pd.DataFrame(physio_dict)
        # get output directory from original file path
        if self.dataset in ["euskalibur", "pan"]:
            assert isinstance(file_orig, str), (
                "Expected file_orig to be a string for euskalibur and pan datasets"
            )
            output_dir = self.file_mapper.get_out_directory(file_orig)
            # get file name from original file path
            file_orig_name = os.path.basename(file_orig)
            # strip 'physio.tsv.gz' from file name
            file_new_name = file_orig_name.rstrip("physio.tsv.gz")
            # add 'desc-preproc_physio.tsv.gz' to file name
            file_new = f"{file_new_name}desc-preproc_physio.tsv.gz"
        elif self.dataset == "nsd":
            if not isinstance(self.file_mapper, FileMapperNSD):
                raise TypeError(
                    f"NSD dataset requires FileMapperNSD, got {type(self.file_mapper)}"
                )
            output_dir = self.file_mapper.data_directory + "/physio/final"
            # parse file components to get session, run and task identifiers
            file_entities = self.file_mapper._parse_func_file_list_components(
                file_list=[file_orig[0]]
            )[0]
            session = file_entities.get("session", "unknownsession")
            run = file_entities.get("run", "unknownrun")
            task = file_entities.get("task", "unknowntask")
            # create file name with session, run and task identifiers
            file_new = (
                f"{self.subject}_task-{task}_session{session}_run{run}_physio.tsv.gz"
            )

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # write out as tsv.gz with pandas
        physio_df.to_csv(
            f"{output_dir}/{file_new}", sep="\t", index=False, compression="gzip"
        )
