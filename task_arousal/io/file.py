"""
Class for iterating over fMRI and physio data files in BIDS format.
"""

import os
import re
import warnings

from glob import glob
from pathlib import Path
from typing import List, Tuple, Literal

from bids import BIDSLayout

from task_arousal.constants import (
    DATA_DIRECTORY_EUSKALIBUR,
    DATA_DIRECTORY_PAN,
    DATA_DIRECTORY_NSD,
    IS_DERIVED,
)


class FileMapperBids:
    """
    Maps file paths for a specific subject's fMRI and physiological data in a BIDS dataset.
    """

    def __init__(self, dataset: Literal["euskalibur", "pan"], subject: str):
        """
        Initialize the FileMapper for a specific subject.

        Parameters
        ----------
        dataset : {'euskalibur', 'pan'}
            The dataset name.
        subject : str
            The subject identifier.
        """
        self.dataset = dataset
        self.subject = subject
        # initialize BIDS layout
        print("Initializing BIDS layout for subject:", subject)
        """
        Note, for EUSKALIBUR dataset: the filemapper class assumes that fmri, physio and event files 
        are in a single BIDS directory structure. Physio files are in the 'raw'
        BIDS directory and are not copied into the FMRIPrep derivatives folder. We
        must manually (or through a script) copy the physio files in the raw BIDS 
        directory to the FMRIPrep derivatives folder to ensure that the FileMapper
        can find them.
        """
        # create ignore pattern for other subjects to speed up layout initialization
        ignore_pattern = re.compile(f"(sub-(?!{self.subject}).*/)")
        # The BIDSLayout initialization can be slow, especially for large datasets
        with warnings.catch_warnings():
            # suppress warnings about soon-to-be-deprecated ignore parameter
            warnings.simplefilter("ignore")
            # Handle Euskalibur preprocessed files, which are placed in a 'derivatives' folder
            # this may need to be adjusted based on how the data is organized
            if self.dataset == "euskalibur":
                if IS_DERIVED:
                    self.layout = BIDSLayout(
                        DATA_DIRECTORY_EUSKALIBUR,
                        is_derivative=True,
                        ignore=[ignore_pattern],
                    )
                else:
                    self.layout = BIDSLayout(
                        DATA_DIRECTORY_EUSKALIBUR,
                        derivatives=True,
                        ignore=[ignore_pattern],
                    )
            elif self.dataset == "pan":
                self.layout = BIDSLayout(
                    DATA_DIRECTORY_PAN,
                    ignore=[ignore_pattern],
                    is_derivative=True,
                    derivatives=True,
                )
            else:
                raise ValueError("Dataset must be 'euskalibur' or 'pan'")

        # get available subjects in the dataset
        self.available_subjects = self.layout.get_subjects()
        # check whether any subjects are found
        if not self.available_subjects:
            raise RuntimeError(
                f"No subjects found in BIDS directory: {
                    DATA_DIRECTORY_EUSKALIBUR
                    if self.dataset == 'euskalibur'
                    else DATA_DIRECTORY_PAN
                }"
            )

        # check if subject is valid
        if subject not in self.available_subjects:
            raise ValueError(f"Subject '{subject}' not found in dataset.")
        # get the sessions for the subject
        self.sessions = self.layout.get_sessions(subject=subject)
        # get the tasks for the subject
        self.tasks = self.layout.get_tasks(subject=subject)
        # loop through sessions and get runs for each task
        self.tasks_runs = {}
        for task in self.tasks:
            self.tasks_runs[task] = {}
            for session in self.sessions:
                self.tasks_runs[task][session] = self.layout.get_runs(
                    subject=subject, session=session, task=task
                )

    def get_fmri_files(
        self,
        task: str,
        sessions: List[str] | None = None,
        preproc_type: Literal["orig", "final"] = "orig",
        func_type: Literal["volume", "surface"] = "volume",
    ) -> list[str]:
        """
        Get the fMRI files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.
        preproc_type : {'orig', 'final'}
            The type of fMRI files to retrieve. 'orig' returns files
            with the 'preproc' description (output of fMRIPrep preprocessing).
            'final' returns files with the 'preprocfinal' description
            (output of additional final preprocessing steps).
        func_type : {'volume', 'surface'}
            The type of functional data. 'volume' returns volumetric files
            with the '.nii.gz' extension. 'surface' returns surface files
            with the '.dtseries.nii' extension.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        # set description and extension based on preprocessing type
        if func_type == "volume":
            extension = ".nii.gz"
            # set desc based on preproc_type
            if preproc_type == "orig":
                desc = "preproc"
            elif preproc_type == "final":
                desc = "preprocfinal"
        elif func_type == "surface":
            extension = ".dtseries.nii"
            # surface files do not have 'preproc' desc in fmripep output
            if preproc_type == "orig":
                desc = None
            elif preproc_type == "final":
                desc = "preprocfinal"
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        fmri_files = []
        for session in sessions if sessions is not None else self.sessions:
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                for run in runs:
                    files = self.get_session_fmri_files(
                        session, task, run=run, desc=desc, extension=extension
                    )
                    fmri_files.extend(files)
            else:
                files = self.get_session_fmri_files(
                    session, task, desc=desc, extension=extension
                )
                fmri_files.extend(files)
        return fmri_files

    def get_physio_files(
        self,
        task: str,
        sessions: List[str] | None = None,
        return_json: bool = False,
        preproc_type: Literal["orig", "final"] = "orig",
    ) -> list[str] | list[tuple[str, str]]:
        """
        Get the physiological files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.
        return_json : bool
            Whether to return the json sidecar files.
        preproc_type : {'orig', 'final'}
            The type of physio files to retrieve. 'orig' returns raw
            'physio' files output from the fMRIPrep pipeline. 'final' returns
            'preproc' physio files that have undergone preprocessing.

        Returns
        -------
        list of str or list of tuple of str
            A list of physiological file paths. If `return_json` is True,
            the physiological file path and JSON sidecar files will be
            returned as a Tuple (physio_file, json_file).
        """
        if preproc_type == "orig":
            desc = None
        elif preproc_type == "final":
            desc = "preproc"
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        physio_files = []
        for session in sessions if sessions is not None else self.sessions:
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                for run in runs:
                    files = self.get_session_physio_files(
                        session, task, run=run, desc=desc
                    )
                    physio_files.extend(files)
            else:
                files = self.get_session_physio_files(session, task, desc=desc)
                physio_files.extend(files)

        return physio_files

    def get_event_files(
        self, task: str, sessions: List[str] | None = None
    ) -> list[list[tuple[str, str]]]:
        """
        Get the event files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        list of list of tuple of str
            A nested list of onset and duration file path tuples (onset, duration) by session.
        """
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        event_files = []
        for session in sessions if sessions is not None else self.sessions:
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                files = []
                for run in runs:
                    run_files = self.get_session_event_files(session, task, run=run)
                    event_files.extend(run_files)
            else:
                files = self.get_session_event_files(session, task)
                event_files.append(files)
        return event_files

    def get_matching_files(
        self,
        file_entities: dict[str, str],
        file_modality: Literal["physio", "fmri"],
        preproc_type: Literal["orig", "final"] = "orig",
        func_type: Literal["volume", "surface"] = "volume",
    ) -> list[str]:
        """
        Get physio, fmri, or event files matching specific BIDS entities.

        Parameters
        ----------
        file_entities : dict of str to str
            A dictionary specifying the BIDS entities to match.
        file_modality : {'physio', 'fmri'}
            The type of files to retrieve. Options are 'physio' for physiological
            files, 'fmri' for fMRI files.
        preproc_type : {'orig', 'final'}
            The stage of processing to retrieve. Options are 'orig' for output of fmriprep files,
            'final' for final files.

        Returns
        -------
        list of str
            A list of matching file paths.
        """
        # determine suffix and extension based on modality
        if file_modality == "physio":
            suffix = "physio"
            extension = ".tsv.gz"
            if preproc_type == "orig":
                desc = None
            elif preproc_type == "final":
                desc = "preproc"

        elif file_modality == "fmri":
            if func_type == "surface":
                suffix = "bold"
                extension = ".dtseries.nii"
            elif func_type == "volume":
                suffix = "bold"
                extension = ".nii.gz"
            if preproc_type == "orig":
                desc = "preproc"
            elif preproc_type == "final":
                desc = "preprocfinal"
        else:
            raise ValueError("file_modality must be 'physio' or 'fmri'")

        # get bid files matching entities
        bids_files = self.layout.get(
            subject=self.subject,
            suffix=suffix,
            extension=extension,
            task=file_entities.get("task", None),
            session=file_entities.get("session", None),
            run=file_entities.get("run", None),
            desc=desc,
            echo=None,
        )
        return [f.path for f in bids_files]

    @staticmethod
    def get_out_directory(fp: str) -> str:
        """
        Get the output directory for a specific file path.

        Parameters
        ----------
        fp : str
            The file path.

        Returns
        -------
        str
            The output directory path.
        """
        return str(Path(fp).parent)

    def get_sessions_task(self, task: str) -> List[str]:
        """
        Get the sessions available for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        list of str
            A list of session identifiers.
        """
        sessions = self.layout.get_sessions(subject=self.subject, task=task)
        return sessions

    def get_session_event_files(
        self, session: str, task: str, run: str | None = None, ped: str | None = None
    ) -> list[tuple[str, str]] | list[str]:
        """
        Get the event files for a specific session and task.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        ped : str, optional
            The phase encoding direction of the fMRI data. If provided, only files
            with this direction will be returned. Options are 'ap' (anterior-posterior)
            and 'pa' (posterior-anterior).

        Returns
        -------
        list of tuple of str | list of str
            A list of onset and duration file path tuples (onset, duration) - for Euskalibur,
            or a list of event file paths - for PAN.
        """
        if self.dataset == "euskalibur":
            return _get_session_event_files_euskalibur(
                self.layout, self.subject, session, task, run=run
            )
        elif self.dataset == "pan":
            return _get_session_event_files_pan(
                self.layout, self.subject, session, task, run=run
            )
        else:
            raise NotImplementedError(
                f"Event file retrieval not implemented for {self.dataset} dataset."
            )

    def get_session_fmri_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        ped: str | None = None,
        desc: Literal["preproc", "preprocfinal"] | None = "preproc",
        extension: str = ".nii.gz",
    ) -> list[str]:
        """
        Get the fMRI files for a specific session and task.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        ped : str, optional
            The phase encoding direction of the fMRI data. If provided, only files
            with this direction will be returned. Options are 'ap' (anterior-posterior)
            and 'pa' (posterior-anterior).
        desc : Literal['preproc', 'preprocfinal'] | None, optional
            The description entity to filter files. Defaults to 'preproc' for
            the output of fMRIPrep preprocessing. Use 'preprocfinal' for
            files that have undergone additional (final) preprocessing steps. Note,
            surface files do not have a description entity in fMRIPrep output,
        extension : str
            The file extension to filter files. Defaults to '.nii.gz' for
            volumetric fMRI files. Use '.dtseries.nii' for surface fMRI files.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        if ped is not None:
            if ped not in ("ap", "pa"):
                raise ValueError("ped must be 'ap' or 'pa'")

        # for some reason, filtering by PhaseEncodingDirection entity does not work for some files
        # so we will filter manually after retrieving the files
        bids_files = self.layout.get(
            subject=self.subject,
            session=session,
            task=task,
            suffix="bold",
            extension=extension,
            run=run,
            desc=desc,
            echo=None,
        )

        if ped is not None:
            filenames = [f.path for f in bids_files if f"dir-{ped}" in f.filename]
        else:
            filenames = [f.path for f in bids_files]

        return filenames

    def get_session_physio_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        desc: Literal["preproc"] | None = None,
    ) -> list[str]:
        """
        Get the physiological file paths for a specific session and task.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        desc : str, optional
            The description entity to filter files. Can provide None to get
            'raw' physio files output from the fMRIPrep pipeline. Can provide
            'preproc' to get physio files that have undergone preprocessing.

        Returns
        -------
        list of str
            A list of physiological file paths.
        """
        bids_files = self.layout.get(
            subject=self.subject,
            session=session,
            task=task,
            suffix="physio",
            extension=".tsv.gz",
            run=run,
            desc=desc,
        )
        filenames = [f.path for f in bids_files]
        return filenames


class FileMapperNSD:
    """
    Maps file paths for a specific subject's fMRI and physiological data in the
    Natural Scene Dataset (NSD). This dataset does not follow BIDS format, so the file paths
    are mapped based on the known directory structure of the NSD dataset rather than using a BIDSLayout.

    Built for API compatibility with FileMapperBids class.
    """

    # hard code subjects in NSD dataset
    available_subjects = [
        "subj01",
        "subj02",
        "subj03",
        "subj04",
        "subj05",
        "subj06",
        "subj07",
        "subj08",
    ]
    # subject label to subject bids translation
    subject_label_to_bids = {
        "subj01": "01",
        "subj02": "02",
        "subj03": "03",
        "subj04": "04",
        "subj05": "05",
        "subj06": "06",
        "subj07": "07",
        "subj08": "08",
    }
    # hard code tasks in NSD dataset
    available_tasks = ["rest", "nsdimagery"]
    # hard code TR by task in NSD dataset (note that TR is not consistent across tasks in NSD, so we cannot set a single constant TR value for the dataset)
    TR_by_task = {"rest": 1.0, "nsdimagery": 1.333}
    # define data directory from constants.py, which can be set through environment variable or defaults to 'data/nsd'
    data_directory = DATA_DIRECTORY_NSD

    def __init__(self, subject: str):
        """
        Initialize the FileMapper for a specific subject.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject

        # check if subject is valid
        if subject not in self.available_subjects:
            raise ValueError(f"Subject '{subject}' not found in dataset.")
        # run a glob search to get all the func_files for that subject
        self.func_files = glob(
            os.path.join(self.data_directory, "func/orig", f"{subject}*.nii.gz")
        )
        # if no files found, raise an error
        if not self.func_files:
            raise RuntimeError(
                f"No functional files found for subject '{subject}' in directory '{self.data_directory}'."
            )
        # parse file components to get sessions and runs
        func_file_components = self._parse_func_file_list_components(self.func_files)
        # extract sessions and runs from file components
        self.sessions = sorted(set(comp["session"] for comp in func_file_components))
        # get the tasks for the subject
        self.tasks = sorted(set(comp["task"] for comp in func_file_components))
        # loop through sessions and get runs for each task
        self.tasks_runs = {}
        for task in self.tasks:
            self.tasks_runs[task] = {}
            for session in self.sessions:
                # filter file components for this session and task
                session_task_comps = [
                    comp
                    for comp in func_file_components
                    if comp["session"] == session and comp["task"] == task
                ]
                # get runs for this session and task
                self.tasks_runs[task][session] = sorted(
                    set(comp["run"] for comp in session_task_comps)
                )

    def get_fmri_files(
        self,
        task: str,
        sessions: List[str] | None = None,
        preproc_type: Literal["orig", "final"] = "orig",
        func_type: Literal["volume", "surface"] = "volume",
    ) -> list[str]:
        """
        Get the fMRI files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.
        preproc_type : {'orig', 'final'}
            The type of fMRI files to retrieve. 'orig' returns the minimally
            preprocessed functional files from NSD.
            'final' returns files from the additional preprocessing steps.
        func_type : {'volume', 'surface'}
            The type of functional data. Only volume files are available for NSD,
            so this parameter does not affect file retrieval but is included
            for API compatibility with FileMapperBids.


        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        # set extension based on preprocessing type
        if func_type == "volume":
            extension = ".nii.gz"
        elif func_type == "surface":
            raise NotImplementedError("Surface files are not available in NSD dataset.")

        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        fmri_files = []
        for session in sessions if sessions is not None else self.sessions:
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                for run in runs:
                    files = self.get_session_fmri_files(
                        session, task, run=run, desc=preproc_type, extension=extension
                    )
                    fmri_files.extend(files)
            else:
                files = self.get_session_fmri_files(
                    session, task, desc=preproc_type, extension=extension
                )
                fmri_files.extend(files)
        return fmri_files

    def get_physio_files(
        self,
        task: str,
        sessions: List[str] | None = None,
        return_json: bool = False,
        preproc_type: Literal["orig", "final"] = "orig",
    ) -> list[Tuple[str, str]] | list[Tuple[Tuple[str, str], str]]:
        """
        Get the physiological files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.
        return_json : bool
            NSD does not provide BIDS-style JSON sidecars for physio files.
            For downstream compatibility with code that expects JSON sidecars, this option allows returning a tuple of
            (physio_file, json_file) where json_file is a placeholder string indicating that no JSON file is available.
        preproc_type : {'orig', 'final'}
            The type of physio files to retrieve. 'orig' returns minimally processed
            'physio' files output from the NSD pipeline. 'final' returns
            'preproc' physio files that have undergone additional preprocessing.

        Returns
        -------
        list of tuple of str
            A list of physiological file paths. If `return_json` is True,
            the physiological file path and JSON sidecar files will be
            returned as a Tuple (physio_file, json_file).
        """

        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        physio_files = []
        for session in sessions if sessions is not None else self.sessions:
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                for run in runs:
                    files = self.get_session_physio_files(
                        session, task, run=run, desc=preproc_type
                    )
                    physio_files.extend(files)
            else:
                files = self.get_session_physio_files(session, task, desc=preproc_type)
                physio_files.extend(files)

        # if return_json is True, return a tuple of (physio_file, json_file) where json_file is a placeholder string indicating that no JSON file is available
        if return_json:
            physio_files = [
                (physio_file, "no_json_available") for physio_file in physio_files
            ]

        return physio_files

    def get_event_files(
        self, task: str, sessions: List[str] | None = None
    ) -> list[list[tuple[str, str]]]:
        """
        Get the event files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        list of list of tuple of str
            A nested list of onset and duration file path tuples (onset, duration) by session.
        """
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        event_files = []
        for session in sessions if sessions is not None else self.sessions:
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                files = []
                for run in runs:
                    run_files = self.get_session_event_files(session, task, run=run)
                    event_files.extend(run_files)
            else:
                files = self.get_session_event_files(session, task)
                event_files.append(files)
        return event_files

    def get_matching_files(
        self,
        file_entities: dict[str, str],
        file_modality: Literal["physio", "fmri"],
        preproc_type: Literal["orig", "final"] = "orig",
        func_type: Literal["volume", "surface"] = "volume",
    ) -> List[str] | list[dict[str, str]]:
        """
        Get matching physio or fmri file paths based on specified entities using
        glob patterns. If an entity is None, it will be treated as a wildcard.

        Note, if file_modality is 'physio' and preproc_type is 'orig', the output
        will a list of dictionaries with keys 'pulse' and 'resp' for the two
        physio files per run (NSD supplieds physio signals as separate files).


        Parameters
        ----------
        file_entities : dict of str to str
            A dictionary specifying the BIDS-like entities to match. Keys can include 'session', 'task', and 'run'.
            Values should be the specific identifier to match, or None to match all
        file_modality : {'physio', 'fmri'}
            The type of files to retrieve. Options are 'physio' for physiological
            files, 'fmri' for fMRI files.
        preproc_type : {'orig', 'final'}
            The stage of processing to retrieve. Options are 'orig' for minimally
             processed files,'final' for final processed files.
        func_type : {'volume', 'surface'}
            The type of functional data. Only volume files are available for NSD,
            so this parameter does not affect file retrieval but is included
            for API compatibility with FileMapperBids.

        Returns
        -------
        list of str
            A list of matching file paths.
        """
        # get session, task and run components
        _session = (
            "*"
            if file_entities.get("session") is None
            else file_entities.get("session")
        )
        _task = "*" if file_entities.get("task") is None else file_entities.get("task")
        _run = "*" if file_entities.get("run") is None else file_entities.get("run")

        # build glob pattern based on modality
        if file_modality == "physio":
            extension = ".tsv"
            # get base directory based on preproc_type
            if preproc_type == "orig":
                base_dir = "physio/orig"
            elif preproc_type == "final":
                base_dir = "physio/final"

            if preproc_type == "orig":
                # for orig physio, return dicts with pulse and resp files
                pulse_pattern = os.path.join(
                    self.data_directory,
                    base_dir,
                    f"{self.subject}_task-{_task}_session{_session}_run{_run}_physio_puls{extension}",
                )
                resp_pattern = os.path.join(
                    self.data_directory,
                    base_dir,
                    f"{self.subject}_task-{_task}_session{_session}_run{_run}_physio_resp{extension}",
                )
                pulse_files = glob(pulse_pattern)
                resp_files = glob(resp_pattern)
                # ensure that the number of pulse and resp files match
                if len(pulse_files) != len(resp_files):
                    raise RuntimeError(
                        "Mismatch in number of pulse and respiration physio files found."
                    )
                # build list of dicts
                fp_out = []
                for p_file, r_file in zip(pulse_files, resp_files):
                    fp_out.append({"pulse": p_file, "resp": r_file})
            else:
                # for final physio, return single files
                pattern = os.path.join(
                    self.data_directory,
                    base_dir,
                    f"{self.subject}_task-{_task}_session{_session}_run{_run}_physio{extension}",
                )
                fp_out = glob(pattern)

        elif file_modality == "fmri":
            # get extension based on func_type
            if func_type == "surface":
                raise NotImplementedError(
                    "Surface files are not available in NSD dataset."
                )
            elif func_type == "volume":
                extension = ".nii.gz"
            # get base directory based on preproc_type
            if preproc_type == "orig":
                base_dir = "func/orig"
            elif preproc_type == "final":
                base_dir = "func/final"

            # build glob pattern
            pattern = os.path.join(
                self.data_directory,
                base_dir,
                f"{self.subject}_task-{_task}_session{_session}_run{_run}{extension}",
            )
            fp_out = glob(pattern)
        else:
            raise ValueError("file_modality must be 'physio' or 'fmri'")

        return fp_out

    @staticmethod
    def get_out_directory(fp: str) -> str:
        """
        Get the output directory for a specific file path.

        Parameters
        ----------
        fp : str
            The file path.

        Returns
        -------
        str
            The output directory path.
        """
        return str(Path(fp).parent)

    def get_sessions_task(self, task: str) -> List[str]:
        """
        Get the sessions available for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        list of str
            A list of session identifiers.
        """
        # get sessions from task_runs dict
        sessions = list(self.tasks_runs[task].keys())
        return sessions

    def get_session_event_files(
        self, session: str, task: str, run: str | None = None, ped: str | None = None
    ) -> list[str]:
        """
        Get NSD event files for a specific session and task. Only the nsdimagery
        task is used currently. These files are in BIDS format.
        See note in the events_to_df method in DatasetNSD class.

        Parameters
        ----------
        session : str
            The session identifier. For the NSD imagery task, a fake session identifier of '01' is used.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned. The NSD imagery
            event files are organized by task (attA, attB, attC) that correspond to runs '02', '05', '08',
            in the functional files, respectively.
        ped : str, optional
            The phase encoding direction of the fMRI data. This parameter is not used
            in the NSD dataset, as event files are not organized by phase encoding direction.
            Kept for compatibility with other FileMapper classes.

        Returns
        -------
        list of str
            A list of BIDS event file paths.
        """
        # run to task mapping for NSD imagery task
        run_task_mapping = {"02": "attA", "05": "attB", "08": "attC"}
        # build path to design directory
        design_dir = os.path.join(self.data_directory, "design")
        # get runs for the NSD imagery task - should always be three runs, but check if run is provided to filter
        if run is not None:
            runs = [run]
        else:
            runs = self.tasks_runs[task][session]

        filenames = []
        for run in runs:
            if task == "nsdimagery":
                # get corresponding task for this run
                task_run = run_task_mapping.get(run, None)
                if task_run is None:
                    raise ValueError(
                        f"Run '{run}' does not correspond to a valid task in NSD imagery dataset."
                    )
                # e.g. sub-01_ses-nsdimagery_session01_task-attA_events.tsv
                pattern = f"sub-{self.subject_label_to_bids[self.subject]}_ses-{task}_session01_task-{task_run}_events.tsv"
                filenames.append(os.path.join(design_dir, pattern))
            else:
                raise ValueError(
                    f"Task '{task}' does not have event files in NSD dataset."
                )

        return filenames

    def get_session_fmri_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        ped: str | None = None,
        desc: Literal["orig", "final"] = "orig",
        extension: str = ".nii.gz",
    ) -> list[str]:
        """
        Get the fMRI files for a specific session and task. Parameters are consistent
        with FileMapperBids for API compatibility. However, the NSD dataset does not
        follow BIDS format, so some parameters are not used and others repurposed.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        ped : str, optional
            The phase encoding direction of the fMRI data. This parameter is not used
            in the NSD dataset, as event files are not organized by phase encoding direction.
            Kept for compatibility with other FileMapper classes.
        desc : Literal['orig', 'final']
            Whether to pull minimally processed ('orig') or fully processed ('final') files.
            Defaults to 'orig'. This parameter is repurposed from description entity
            filtering in BIDS datasets to indicate processing stage in NSD dataset.
        extension : str
            The file extension to filter files. Defaults to '.nii.gz' for
            volumetric fMRI files. Use '.dtseries.nii' for surface fMRI files.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        # build path to minimally preprocessed or fully processed func directory
        if desc == "orig":
            func_dir = os.path.join(self.data_directory, "func/orig")
        elif desc == "final":
            func_dir = os.path.join(self.data_directory, "func/final")
        else:
            raise ValueError("desc must be 'orig' or 'final'")
        # get runs for this session and task - NSD always multiple runs for resting-state
        if run is not None:
            runs = [run]
        else:
            runs = self.tasks_runs[task][session]
        # construct file path pattern
        filenames = []
        for run in runs:
            if desc == "orig":
                pattern = (
                    f"{self.subject}_task-{task}_session{session}_run{run}{extension}"
                )
            elif desc == "final":
                pattern = f"{self.subject}_task-{task}_session{session}_run{run}_desc-preprocfinal{extension}"
            filenames.append(os.path.join(func_dir, pattern))
        return filenames

    def get_session_physio_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        desc: Literal["orig", "final"] = "orig",
    ) -> list[str] | list[Tuple[str, str]]:
        """
        Get the physiological file paths for a specific session and task. Parameters are consistent
        with FileMapperBids for API compatibility. However, the NSD dataset does not
        follow BIDS format, so some parameters are not used and others repurposed.

        Note, respiratory and pulse data are stored separately in NSD dataset. If the desc = 'orig',
        both files will be returned for each run as a list of tuples with pulse signals in the first position and
        respiratory signals in the second position. Both signals are combined into a single physio file in the 'final'
        processing stage. If the desc='final', only a single physio file will be returned for each run.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        desc : Literal["orig", "final"]
            Whether to pull minimally processed ('orig') or fully processed ('final') files.
            Defaults to 'orig'. This parameter is repurposed from description entity
            filtering in BIDS datasets to indicate processing stage in NSD dataset.

        Returns
        -------
        list of str | list[Tuple[str, str]]
            A list of physiological file paths (desc='final') or
            a list of tuples with pulse signals in the first position and
            respiratory signals in the second position (desc='orig').
        """
        # get build path to minimally preprocessed or fully processed physio directory
        if desc == "orig":
            physio_dir = os.path.join(self.data_directory, "physio/orig")
        elif desc == "final":
            physio_dir = os.path.join(self.data_directory, "physio/final")
        else:
            raise ValueError("desc must be 'orig' or 'final'")
        # get runs for this session and task - NSD always multiple runs
        # if run is provided, use that, otherwise get all runs for this session and task
        if run is not None:
            runs = [run]
        else:
            runs = self.tasks_runs[task][session]
        # construct file path pattern
        filenames = []
        for run in runs:
            if desc == "orig":
                resp_pattern = (
                    f"{self.subject}_task-{task}_session{session}_run{run}_resp.tsv"
                )
                pulse_pattern = (
                    f"{self.subject}_task-{task}_session{session}_run{run}_puls.tsv"
                )
                # first, check that the resp and pulse files exist for this run
                resp_fp = os.path.join(physio_dir, resp_pattern)
                pulse_fp = os.path.join(physio_dir, pulse_pattern)
                # if path doesn't exist, don't raise error - just skip this run and move on to the next one,
                # since some runs are  missing physio data in NSD dataset
                if not os.path.exists(resp_fp) or not os.path.exists(pulse_fp):
                    print(
                        f"Physio data missing for run {run} in session {session} for subject {self.subject}. Skipping this run."
                    )
                    continue
                filenames.append(
                    (
                        os.path.join(physio_dir, pulse_pattern),
                        os.path.join(physio_dir, resp_pattern),
                    )
                )
            elif desc == "final":
                physio_pattern = f"{self.subject}_task-{task}_session{session}_run{run}_physio.tsv.gz"
                # . check that the physio file exists for this run
                physio_fp = os.path.join(physio_dir, physio_pattern)
                if not os.path.exists(physio_fp):
                    print(
                        f"Physio data missing for run {run} in session {session} for subject {self.subject}. Skipping this run."
                    )
                    continue
                filenames.append(os.path.join(physio_dir, physio_pattern))

        return filenames

    def get_subject_mask(self):
        """
        Get the subject functional mask file path. NSD dataset provides a brain mask for each subject.

        Returns
        -------
        str
            The file path to the subject's brain mask.
        """
        mask_path = os.path.join(
            self.data_directory, "mask", f"{self.subject}_func1pt8mm_brain_mask.nii.gz"
        )
        if not os.path.exists(mask_path):
            raise RuntimeError(
                f"Mask file not found for subject '{self.subject}' at path '{mask_path}'."
            )
        return mask_path

    def get_tr(self, task: str) -> float:
        """
        Get the TR for a specific task in the NSD dataset. TR is not consistent across tasks in NSD, so we cannot set a single constant TR value for the dataset.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        float
            The TR for the specified task.
        """
        if task not in self.TR_by_task:
            raise ValueError(f"Task '{task}' not found in dataset.")
        return self.TR_by_task[task]

    def _parse_func_file_list_components(
        self, file_list: list[str]
    ) -> list[dict[str, str]]:
        """
        Take the results of a glob search in the functional directory and parse out the subject, session, task, and
        run components of the file paths.

        Expected functional file path format:
        <subject>_task-<task>_session<session>_run<run>.nii.gz
        """
        fp_components = []
        for fp in file_list:
            # get filename from file path
            filename = os.path.basename(fp)
            # first parse by underscores to separate entities
            components = filename.split("_")
            # get subject
            subject = components[0]
            # get task
            task = components[1].split("-")[1]
            # get session
            session = components[2].split("session")[1]
            # get run
            run = components[3].split("run")[1].split(".nii.gz")[0]
            fp_components.append(
                {"subject": subject, "task": task, "session": session, "run": run}
            )
            # check that subject matches the FileMapper's subject
            if subject != self.subject:
                raise ValueError(
                    f"File path '{fp}' subject '{subject}' does not match FileMapper subject '{self.subject}'."
                )
            # check that task is valid for the subject
            if task not in self.available_tasks:
                raise ValueError(
                    f"File path '{fp}' task '{task}' is not valid for subject '{self.subject}'."
                )
            # try to convert session and run to an integer to check that they are valid integers
            try:
                int(session)
                int(run)
            except ValueError:
                raise ValueError(
                    f"File path '{fp}' session '{session}' or run '{run}' is not a valid integer."
                )
            # check that session and run are present
            if not all([session, run]):
                raise ValueError(f"File path '{fp}' does not contain a session or run.")
        return fp_components


def get_dataset_subjects(dataset: str) -> List[str]:
    """
    Get the list of available subjects for a specific dataset.

    Parameters
    ----------
    dataset : str
        The dataset name. Options are 'euskalibur', 'pan', or 'nsd'.

    Returns
    -------
    list of str
        A list of subject identifiers
    """
    if dataset == "euskalibur":
        if IS_DERIVED:
            layout = BIDSLayout(
                DATA_DIRECTORY_EUSKALIBUR,
                is_derivative=True,
            )
        else:
            layout = BIDSLayout(
                DATA_DIRECTORY_EUSKALIBUR,
                derivatives=True,
            )
        subjects = layout.get_subjects()
    elif dataset == "pan":
        layout = BIDSLayout(
            DATA_DIRECTORY_PAN,
            is_derivative=True,
            derivatives=True,
        )
        subjects = layout.get_subjects()
    elif dataset == "nsd":
        subjects = FileMapperNSD.available_subjects
    else:
        raise ValueError("Dataset must be 'euskalibur', 'pan', or 'nsd'.")

    return subjects


def _get_session_event_files_euskalibur(
    layout: BIDSLayout, subject: str, session: str, task: str, run: str | None = None
) -> list[tuple[str, str]]:
    """
    Get the event files for a specific session and task for EuskalIBUR dataset.
    Event files include task onset and duration 1D files output from AFNI preprocessing.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDS layout object.
    subject : str
        The subject identifier.
    session : str
        The session identifier.
    task : str
        The task identifier.
    run : str, optional
        The run identifier. If provided, only files for this run will be returned.

    Returns
    -------
    list of tuple of str
        A list of onset and duration file path tuples (onset, duration).
    """
    bids_files_onset = layout.get(
        subject=subject,
        session=session,
        task=task,
        suffix="onset",
        extension=".1D",
        run=run,
    )
    fp_onsets = [f.path for f in bids_files_onset]
    # for the simon task, no duration files are provided from congruent/incongruent events
    # so we will return None for duration files
    if task == "simon":
        fp_durations = [""] * len(bids_files_onset)
    else:
        bids_files_duration = layout.get(
            subject=subject,
            session=session,
            task=task,
            suffix="duration",
            extension=".1D",
            run=run,
        )
        fp_durations = [f.path for f in bids_files_duration]
    return list(zip(fp_onsets, fp_durations))


def _get_session_event_files_pan(
    layout: BIDSLayout, subject: str, session: str, task: str, run: str | None = None
) -> list[str]:
    """
    Get the event files for a specific session and task for PAN dataset. Event files
    are in BIDS format.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDS layout object.
    subject : str
        The subject identifier.
    session : str
        The session identifier.
    task : str
        The task identifier.
    run : str, optional
        The run identifier. If provided, only files for this run will be returned.

    Returns
    -------
    list of str
        A list of event file paths.
    """
    ev_files = layout.get(
        subject=subject, session=session, task=task, extension=".1D", run=run
    )
    fp_events = [f.path for f in ev_files]

    return fp_events
