"""
Preprocessing pipeline for fMRI and physiological data.

fMRI volume or surface preprocessing is performed on outputs from fMRIPrep for EuskalIBUR and
Natural Scenes Dataset, including:

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
import shutil
import warnings

from typing import Literal, Tuple, List

import nibabel as nib
import numpy as np
import pandas as pd

# dataset metadata
from task_arousal.constants import (
    MASK_EUSKALIBUR,  # brain mask for EuskalIBUR
    # brain mask for NSD is subject-specific and retrieved from file mapper rather than as a constant
    PHYSIO_COLUMNS_EUSKALIBUR,  # physio columns to extract for EuskalIBUR
    PHYSIO_COLUMNS_NSD,  # physio columns to extract for NSD
    TR_EUSKALIBUR,  # TR for EuskalIBUR
    # TR for NSD is handled in the file mapper class since it differs by task
    SLICE_TIMING_REF,  # slice timing reference
    ECHOS_EUSKALIBUR,  # echo times for EuskalIBUR
)

## Preprocessing parameters
from task_arousal.constants import (
    DUMMY_VOLUMES,  # number of dummy volumes to drop
    HIGHPASS,  # high-pass filter cutoff frequency
    FWHM_EUSKALIBUR,  # smoothing FWHM for EuskalIBUR
    FWHM_NSD,  # smoothing FWHM for NSD
    PHYSIO_RESAMPLE_F,  # physio resample frequency
    SURFACE_LH,  # left hemisphere surface template
    SURFACE_RH,  # right hemisphere surface template
)

from task_arousal.io.file import FileMapperBids, FileMapperNSD
from task_arousal.preprocess.components.multiecho_fit import (
    fit_multiecho,
    multiecho_to_std,
)
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
        dataset: Literal["euskalibur", "nsd"],
        subject: str,
    ) -> None:
        """Initialize the preprocessing pipeline for a specific dataset and subject.

        Parameters
        ----------
            dataset (Literal['euskalibur', 'nsd']): The dataset identifier.
            subject (str): The subject identifier.
            func_type (Literal['volume', 'surface'], optional): The type of functional data. Defaults to "volume".
        """
        self.subject = subject
        self.dataset = dataset

        # map file paths associated to subject
        if dataset == "euskalibur":
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
        func_type: Literal["volume", "surface"] = "volume",
        sessions: list[str] | None = None,
        skip_func: bool = False,
        skip_physio: bool = False,
        me_type: List[Literal["optcomb", "t2", "s0"]] = ["optcomb"],
        save_physio_figs: bool = False,
        echo_pipeline: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Preprocess fMRI and physiological data.

        Note, for surface-based preprocessing, you must have connectome workbench installed and configured on your system.

        Parameters
        ----------
        task : str or None, optional
            The task identifier. If no task identifier is provided, all tasks will be processed. Defaults to None.
        func_type : Literal['volume', 'surface'], optional
            The type of functional data to preprocess - volume-preprocessing or surface-based preprocessing.
            Defaults to "volume".
        sessions : list of str or None, optional
            The session identifiers. If no session identifiers are provided, all sessions will be processed. Defaults to None.
        skip_func : bool, optional
            Whether to skip fMRI preprocessing. Defaults to False.
        skip_physio : bool, optional
            Whether to skip physiological preprocessing. Defaults to False.
        me_type: List[Literal['optcomb', 't2', 's0']], optional
            The type of multi-echo functional file to retrieve and process. Only for multi-echo
            datasets. Can be multiple. This parameter is ignored for single-echo datasets. 'optcomb' returns the optimally combined
            multi-echo functional files. 't2' returns the T2* map time series estimated from the multi-echo data.
            's0' returns the S0 map time series estimated from the multi-echo data. Defaults to ['optcomb']. If
            't2' or 's0' is selected, the echo_pipeline must be set to True or previously run to generate the T2* and S0 maps.
        echo_pipeline : bool, optional
            For the Euskalibut dataset. Whether to estimate T2* and S0 from multi-echo fMRI data using a log-linear fit
            and use the estimated T2* and S0 values for preprocessing instead of the raw echo data. Defaults to False.
        """
        # check workbench is available in the system for surface-based preprocessing
        if func_type == "surface":
            if not shutil.which("wb_command"):
                raise RuntimeError(
                    "Workbench command line tool 'wb_command' not found. Surface-based preprocessing requires workbench to be installed and configured on your system."
                )
        # check that ANTS is available in the system for multi-echo fitting
        if echo_pipeline:
            if not shutil.which("antsApplyTransforms"):
                raise RuntimeError(
                    "ANTs command line tool 'antsApplyTransforms' not found. Multi-echo fitting requires ANTs to be installed and configured on your system."
                )

        # parameter checks and dataset-specific handling
        if self.dataset == "nsd":
            # the echo pipeline and surface-based preprocessing are not applicable to NSD since there are no multi-echo data or surface files in NSD,
            # so we raise an error if the user tries to apply these preprocessing steps to NSD. Additionally,
            # there is no physiological data for the nsdimagery task in NSD, so we skip physio preprocessing for that task and
            # warn the user if they try to apply physio preprocessing to that task.
            if echo_pipeline:
                warnings.warn(
                    "Echo pipeline is not applicable to NSD dataset since there are no multi-echo data, so echo_pipeline parameter will be ignored."
                )
            if func_type == "surface":
                raise ValueError(
                    "Surface-based preprocessing is not applicable to NSD dataset since there are no surface files, so func_type cannot be set to 'surface' for NSD."
                )
            if task == "nsdimagery":
                if skip_physio:
                    warnings.warn(
                        "Skipping physiological preprocessing for this task in NSD dataset has no effect - no physio data for nsdimagery task."
                    )
                skip_physio = True
        # check that echo_pipeline is not applied to surface files -
        # individual echo files are not available in surface format
        if echo_pipeline and func_type == "surface":
            raise ValueError(
                "Echo pipeline cannot be applied to surface files since individual echos are not available."
            )
        # currently t2 and s0 map preprocessing (after estimation) is only implemented for volume files,
        # so we raise an error if the user tries to apply t2 or s0 map preprocessing to surface files
        # TODO: implement t2 and s0 map preprocessing for surface files in the future, which would involve
        # mapping the t2 and s0 estimates to the surface and then applying the surface-based preprocessing
        # steps using the mapped t2 and s0 values instead of the raw functional data.
        if ("t2" in me_type or "s0" in me_type) and func_type == "surface":
            raise ValueError(
                "T2 and S0 map preprocessing cannot be applied to surface files since it is only implemented for volume files."
            )
        # check that both func and physio are not skipped
        if skip_func and skip_physio:
            raise ValueError(
                "Both fMRI and physiological preprocessing cannot be skipped."
            )

        if skip_func and verbose:
            print(f"Skipping fMRI preprocessing for subject '{self.subject}'...")
        if skip_physio and verbose:
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
            elif self.dataset == "nsd":
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
                # multi-echo pipeline only applicable to EuskalIBUR since NSD is a single-echo dataset
                if echo_pipeline and self.dataset == "euskalibur":
                    if verbose:
                        print(
                            "Applying multi-echo preprocessing pipeline to estimate T2* and S0 maps"
                        )
                    # estimate t2* and s0 maps from multi-echo data
                    self._multiecho_pipeline(
                        task=task_proc,
                        sessions=sessions,
                        echo_times=ECHOS_EUSKALIBUR,
                        verbose=verbose,
                    )
                    assert isinstance(self.file_mapper, FileMapperBids)
                    self.file_mapper.refresh_layout()
                # loop through multi-echo types (for multi-echo datasets) or just once for single-echo datasets
                for me in me_type:
                    # print which multi-echo type is being processed
                    if self.dataset == "euskalibur" and me_type != ["optcomb"]:
                        print(f"Processing multi-echo type: {me}")

                    # get fmri files for task
                    fmri_files = self.file_mapper.get_fmri_files(
                        task_proc, sessions=sessions, func_type=func_type, me_type=me
                    )
                    # loop through fmri files and preprocess
                    for fmri_file in fmri_files:
                        if verbose:
                            print(f"Preprocessing fMRI file: {fmri_file}")

                        # Apply the functional MRI preprocessing pipeline
                        if func_type == "volume":
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
                        elif func_type == "surface":
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
                            raise ValueError(f"Unknown functional type: {func_type}")
                        # Write out the preprocessed fMRI file
                        self.write_out_fmri_file(
                            fmri_proc,
                            fmri_file,
                            func_type=func_type,
                            me_type=me
                            if (self.dataset == "euskalibur") and (me in ["t2", "s0"])
                            else None,  # pass me_type for euskalibur multi-echo data, but not for NSD since it is single-echo
                        )

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
        func_type: Literal["volume", "surface"] = "volume",
        me_type: str | None = None,
    ) -> None:
        """Write out the preprocessed fMRI file.

        Parameters
        ----------
        fmri_img : nib.nifti1.Nifti1Image | nib.cifti2.cifti2.Cifti2Image
            The preprocessed fMRI image.
        file_orig : str
            The original file path
        func_type : Literal['volume', 'surface'], optional
            The type of functional data. Defaults to "volume".
        me_type : str | None, optional
            The type of multi-echo data. Defaults to None.

        """
        # get output directory from original file path
        # for the Euskalbur dataset, preprocessed files are saved in the
        # same directory as original files. For NSD, preprocessed files are saved in a separate directory.
        if self.dataset == "euskalibur":
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

        def _make_desc_preprocfinal_name(
            name: str, me_type: str | None, out_ext: str
        ) -> str:
            """Create a BIDS-ish output filename with `desc-preprocfinal`."""
            base = _strip_known_fmri_extensions(name)
            if me_type in ["t2", "s0"]:
                me_ext = me_type
                source_desc = f"desc-preproc{me_type}"
            else:
                me_ext = ""
                source_desc = "desc-preproc"

            if "desc-preprocfinal" in base:
                new_base = base
            elif "desc-preproc" in base:
                new_base = base.replace(source_desc, f"desc-preprocfinal{me_ext}", 1)
            elif "_bold" in base:
                new_base = base.replace("_bold", f"_desc-preprocfinal{me_ext}_bold", 1)
            else:
                new_base = f"{base}_desc-preprocfinal{me_ext}"

            return f"{new_base}{out_ext}"

        if func_type == "volume":
            # NIfTI outputs are always `.nii.gz` in this project.
            output_name = _make_desc_preprocfinal_name(
                file_orig_name, me_type, ".nii.gz"
            )
            output_path = f"{output_dir}/{output_name}"

            if not isinstance(fmri_img, nib.nifti1.Nifti1Image):
                raise TypeError(
                    f"Expected NIfTI image for volume output, got {type(fmri_img)}"
                )
            nib.nifti1.save(fmri_img, output_path)

        elif func_type == "surface":
            output_name = _make_desc_preprocfinal_name(
                file_orig_name, me_type, ".dtseries.nii"
            )
            output_path = f"{output_dir}/{output_name}"

            if not isinstance(fmri_img, nib.cifti2.cifti2.Cifti2Image):
                raise TypeError(
                    f"Expected CIFTI image for surface output, got {type(fmri_img)}"
                )
            fmri_img.to_filename(output_path)

        else:
            raise ValueError(f"Unknown functional type: {func_type}")

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
        if self.dataset == "euskalibur":
            assert isinstance(file_orig, str), (
                "Expected file_orig to be a string for euskalibur dataset"
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

    def _multiecho_pipeline(
        self,
        task: str,
        echo_times: List[float],
        sessions: List[str] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Apply multi-echo preprocessing pipeline to estimate T2* and S0 from multi-echo fMRI data using a log-linear fit
        and use the estimated T2* and S0 values for preprocessing instead of the raw echo data.
        This pipeline is only applicable to the EuskalIBUR dataset, which has multi-echo fMRI data.

        Parameters
        ----------
        task : str
            The task identifier for which to apply the multi-echo pipeline.
        echo_times : List[float]
            The echo times (in milliseconds) corresponding to the multi-echo fMRI data.
        sessions : List[str] | None, optional
            The session identifiers to include. If None, all sessions will be included. Defaults to None
        verbose : bool, optional
            Whether to print verbose output during processing. Defaults to True.
        """
        # get fmri files for each echo for task
        fmri_files = self.file_mapper.get_echo_files(
            task,
            sessions=sessions,
        )
        # loop through scans and apply multi-echo fit to estimate T2* and S0 maps and time series
        # and transform to standard space
        for scan_files in fmri_files:
            if verbose:
                print(f"Estimating T2* and S0 maps for fMRI file: {scan_files[0]}")
            # fmri files should a nested list of echo files, where the top level list is by scan and the second level list is by echo.
            # We check that the files are in the expected format here.
            if not isinstance(scan_files, list):
                raise TypeError(
                    f"Expected list of echo files for each scan, got {type(scan_files)}"
                )
            for echo_file in scan_files:
                if not isinstance(echo_file, str):
                    raise TypeError(
                        f"Expected string file path for each echo, got {type(echo_file)}"
                    )
            # finding files for this pipeline useses BIDs entities parsed from the first echo file
            # TODO: this method is only available in PyBIDS layouts, so if a new file mapper is added that does not use a PyBIDS layout,
            # we will need to implement a different method for finding the matching functional mask
            scan_file_ents = self.file_mapper.layout.parse_file_entities(scan_files[0])  # type: ignore
            mask_fp = self.file_mapper.get_subject_mask(
                task=task,
                session=scan_file_ents.get("session"),
                run=scan_file_ents.get("run"),
            )
            # fit t2* and s0 maps using log-linear fit across echoes
            t2_img, s0_img = fit_multiecho(
                fp_echos=scan_files, echo_times=echo_times, mask_fp=mask_fp
            )
            assert isinstance(t2_img, nib.nifti1.Nifti1Image), (
                f"Expected T2* image to be a NIfTI image, got {type(t2_img)}"
            )
            assert isinstance(s0_img, nib.nifti1.Nifti1Image), (
                f"Expected S0 image to be a NIfTI image, got {type(s0_img)}"
            )
            # get files necessary for transforming t2* and s0 maps to standard space using antsApplyTransforms
            std_ref, native_to_t1, t1_to_std = self._find_std_transform_files(
                scan_file_ents
            )
            # ANTs only accepts file paths, so the helper round-trips these images
            # through a temporary directory and returns the transformed images.
            t2_img_std = multiecho_to_std(
                img=t2_img,
                std_space_ref_fp=std_ref,
                native_to_t1w_fp=native_to_t1,
                t1w_to_std_fp=t1_to_std,
                output_fp=None,
            )
            s0_img_std = multiecho_to_std(
                img=s0_img,
                std_space_ref_fp=std_ref,
                native_to_t1w_fp=native_to_t1,
                t1w_to_std_fp=t1_to_std,
                output_fp=None,
            )
            # save out the transformed T2* and S0 images to the same directory as the original echo files with a modified file name indicating that they are T2* and S0 maps
            t2_output_fp = (
                scan_files[0]
                .replace(
                    "_echo-1",
                    "_space-MNI152NLin2009cAsym",  # first, remove the echo identifier
                )
                .replace(
                    "desc-preproc",
                    "desc-preproct2",  # then add the t2 identifier to the descriptor
                )
            )
            s0_output_fp = (
                scan_files[0]
                .replace(
                    "_echo-1",
                    "_space-MNI152NLin2009cAsym",  # first, remove the echo identifier
                )
                .replace(
                    "desc-preproc",
                    "desc-preprocs0",  # then add the s0 identifier to the descriptor
                )
            )
            nib.nifti1.save(t2_img_std, t2_output_fp)
            nib.nifti1.save(s0_img_std, s0_output_fp)

    def _find_std_transform_files(
        self, file_ents: dict[str, str]
    ) -> Tuple[str, str, str]:
        """Find the standard space transform files (e.g. ANTs .h5 files) for a given session and run.

        This method is used in the multi-echo pipeline to find the necessary transform files for
        transforming the estimated T2* and S0 maps to standard space using antsApplyTransforms.

        Files searched for:
        1) The standard space bold reference image (e.g. MNI152NLin2009cAsym) that is the target of the standard space transform.
        2) The native-to-T1 transform file (.txt file) that transforms from the native space of the functional data to the T1 anatomical space.
        3) The T1-to-standard transform file (e.g. ANTs .h5 file) that transforms from the T1 anatomical space to the standard space.

        Parameters
        ----------
        file_ents : dict[str, str]
            A dictionary containing the file entities (e.g., session, run) for the functional data.

        Returns
        -------
        Tuple[str, str, str]
            A tuple containing the file paths for the standard reference image,
            native-to-T1 transform, and T1-to-standard transform, respectively.
        """

        # standard reference image
        std_ref = self.file_mapper.layout.get(  # type: ignore
            subject=self.subject,
            task=file_ents.get("task"),
            session=file_ents.get("session"),
            space="MNI152NLin2009cAsym",
            run=file_ents.get("run"),
            suffix="boldref",
        )
        if std_ref is None or len(std_ref) == 0:
            raise FileNotFoundError(
                f"Standard reference image not found for subject '{self.subject}' in file mapper layout."
            )
        elif len(std_ref) > 1:
            raise ValueError(
                f"Multiple standard reference images found for subject '{self.subject}' in file mapper layout. Expected exactly one."
            )
        else:
            std_ref = std_ref[0].path
        # native to T1 image transform (should be the same for all echoes since they are all in the same native space)
        native_to_t1 = self.file_mapper.layout.get(  # type: ignore
            subject=self.subject,
            task=file_ents.get("task"),
            to="T1w",
            mode="image",
            session=file_ents.get("session"),
            run=file_ents.get("run"),
            desc="coreg",
            extension="txt",
        )
        if native_to_t1 is None or len(native_to_t1) == 0:
            raise FileNotFoundError(
                f"Native to T1 transform not found for subject '{self.subject}' in file mapper layout."
            )
        elif len(native_to_t1) > 1:
            raise ValueError(
                f"Multiple native to T1 transforms found for subject '{self.subject}' in file mapper layout. Expected exactly one."
            )
        else:
            native_to_t1 = native_to_t1[0].path

        # T1 to standard transform (in anat directory)
        t1_to_std = self.file_mapper.layout.get(  # type: ignore
            subject=self.subject,
            to="MNI152NLin2009cAsym",
            mode="image",
        )
        if t1_to_std is None or len(t1_to_std) == 0:
            raise FileNotFoundError(
                f"T1 to standard transform not found for subject '{self.subject}' in file mapper layout."
            )
        elif len(t1_to_std) > 1:
            raise ValueError(
                f"Multiple T1 to standard transforms found for subject '{self.subject}' in file mapper layout. Expected exactly one."
            )
        else:
            t1_to_std = t1_to_std[0].path

        return std_ref, native_to_t1, t1_to_std
