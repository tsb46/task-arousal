"""
Class for iterating over Euskalibur and HCP fMRI and physio data files
"""
import os
import re
import warnings

from glob import glob
from pathlib import Path
from typing import List, Tuple, Literal

import pandas as pd

from bids import BIDSLayout

from task_arousal.constants import DATA_DIRECTORY_EUSKALIBUR, DATA_DIRECTORY_HCP, IS_DERIVED


class FileMapperEuskalibur:
    """
    Maps file paths for a specific subject's fMRI and physiological data in the
    Euskalibur BIDS dataset.
    """
    # specify dataset name
    DATASET = 'euskalibur'
    
    def __init__(self, subject: str):
        """
        Initialize the FileMapper for a specific subject.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # initialize BIDS layout
        print("Initializing BIDS layout for subject:", subject)
        """
        Note: the filemapper class assumes that fmri, physio and event files 
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
            if IS_DERIVED:
                self.layout = BIDSLayout(DATA_DIRECTORY_EUSKALIBUR, is_derivative=True, ignore=[ignore_pattern])
            else:
                self.layout = BIDSLayout(DATA_DIRECTORY_EUSKALIBUR, derivatives=True, ignore=[ignore_pattern])

        # get available subjects in the dataset
        self.available_subjects = self.layout.get_subjects()
        # check whether any subjects are found
        if not self.available_subjects:
            raise RuntimeError(f"No subjects found in BIDS directory: {DATA_DIRECTORY_EUSKALIBUR}")

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
        file_type: Literal['fmriprep', 'final'] = 'fmriprep'
    ) -> list[str]:
        """
        Get the fMRI files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.
        file_type : {'fmriprep', 'final'}
            The type of fMRI files to retrieve. 'fmriprep' returns files
            with the 'preproc' description (output of fMRIPrep preprocessing).
            'final' returns files with the 'preprocfinal' description
            (output of additional final preprocessing steps).

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        if file_type == 'fmriprep':
            desc = 'preproc'
        elif file_type == 'final':
            desc = 'preprocfinal'
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(f"Session '{session}' is not valid for subject '{self.subject}'.")

        fmri_files = []
        for session in (sessions if sessions is not None else self.sessions):
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                for run in runs:
                    files = self.get_session_fmri_files(session, task, run=run, desc=desc)
                    fmri_files.extend(files)
            else:
                files = self.get_session_fmri_files(session, task, desc=desc)
                fmri_files.extend(files)
        return fmri_files

    def get_physio_files(
        self, 
        task: str,
        sessions: List[str] | None = None,
        return_json: bool = False,
        file_type: Literal['fmriprep', 'final'] = 'fmriprep'
    ) -> list[str] | list[Tuple[str,str]]:
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
        file_type : {'fmriprep', 'final'}
            The type of physio files to retrieve. 'fmriprep' returns raw 
            'physio' files output from the fMRIPrep pipeline. 'final' returns
            'preproc' physio files that have undergone preprocessing.

        Returns
        -------
        list of str or list of tuple of str
            A list of physiological file paths. If `return_json` is True,
            the physiological file path and JSON sidecar files will be
            returned as a Tuple (physio_file, json_file).
        """
        if file_type == 'fmriprep':
            desc = None
        elif file_type == 'final':
            desc = 'preproc'
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(f"Session '{session}' is not valid for subject '{self.subject}'.")

        physio_files = []
        for session in (sessions if sessions is not None else self.sessions):
            # check for multiple runs
            runs = self.tasks_runs[task][session]
            # if multiple runs, loop through and get files for each run
            if len(runs) > 1:
                for run in runs:
                    files = self.get_session_physio_files(session, task, run=run, desc=desc)
                    physio_files.extend(files)
            else:
                files = self.get_session_physio_files(session, task, desc=desc)
                physio_files.extend(files)
        if return_json:
            return [(f, f.replace('.tsv.gz', '.json')) for f in physio_files]
        return physio_files

    def get_event_files(self, task: str, sessions: List[str] | None = None) -> list[list[tuple[str, str]]]:
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
                    raise ValueError(f"Session '{session}' is not valid for subject '{self.subject}'.")

        event_files = []
        for session in (sessions if sessions is not None else self.sessions):
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
        file_entities: dict[str,str], 
        file_modality: Literal['physio','fmri'],
        file_type: Literal['fmriprep', 'final'] = 'fmriprep'
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
        file_type : {'fmriprep', 'final'}
            The stage of processing to retrieve. Options are 'fmriprep' for raw files,
            'final' for final files.

        Returns
        -------
        list of str
            A list of matching file paths.
        """
        # determine suffix and extension based on modality
        if file_modality == 'physio':
            suffix = 'physio'
            extension = '.tsv.gz'
            if file_type == 'fmriprep':
                desc = None
            elif file_type == 'final':
                desc = 'preproc'

        elif file_modality == 'fmri':
            suffix = 'bold'
            extension = '.nii.gz'
            if file_type == 'fmriprep':
                desc = 'preproc'
            elif file_type == 'final':
                desc = 'preprocfinal'
        else:
            raise ValueError("file_modality must be 'physio' or 'fmri'")

        # get bid files matching entities
        bids_files = self.layout.get(
            subject=self.subject, 
            suffix=suffix, 
            extension=extension,
            task=file_entities.get('task', None),
            session=file_entities.get('session', None),
            run=file_entities.get('run', None),
            desc=desc,
            echo=None
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

    def get_session_fmri_files(
        self, 
        session: str, 
        task: str, 
        run: str | None = None,
        desc:  Literal['preproc', 'preprocfinal'] = 'preproc'
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
        desc : Literal['preproc', 'preprocfinal']
            The description entity to filter files. Defaults to 'preproc' for 
            the output of fMRIPrep preprocessing. Use 'preprocfinal' for 
            files that have undergone additional (final) preprocessing steps.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        bids_files = self.layout.get(
            subject=self.subject, session=session, task=task, suffix='bold', extension='.nii.gz',
            run=run, desc=desc, echo=None
        )
        filenames = [f.path for f in bids_files]
        return filenames

    def get_session_physio_files(
        self, 
        session: str, 
        task: str, 
        run: str | None = None,
        desc: Literal['preproc'] | None = None
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
            subject=self.subject, session=session, task=task, suffix='physio', extension='.tsv.gz',
            run=run, desc=desc
        )
        filenames = [f.path for f in bids_files]
        return filenames

    def get_session_event_files(
        self, 
        session: str, 
        task: str, 
        run: str | None = None
    ) -> list[tuple[str, str]]:
        """
        Get the event files for a specific session and task. Event files
        include task onset and duration 1D files output from AFNI preprocessing.

        Parameters
        ----------
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
        bids_files_onset = self.layout.get(
            subject=self.subject, session=session, task=task, suffix='onset', extension='.1D',
            run=run
        )
        fp_onsets = [f.path for f in bids_files_onset]
        # for the simon task, no duration files are provided from congruent/incongruent events
        # so we will return None for duration files
        if task == 'simon':
            fp_durations = [''] * len(bids_files_onset)
        else:
            bids_files_duration = self.layout.get(
                subject=self.subject, session=session, task=task, suffix='duration', extension='.1D',
                run=run
            )
            fp_durations = [f.path for f in bids_files_duration]
        return list(zip(fp_onsets, fp_durations))


class FileMapperHCP:
    """
    Maps file paths for a specific subject's fMRI and physiological data in the
    HCP dataset (Human Connectome Project). HCP data is not in BIDS format, so
    this class requires different handling.
    """
    # specify dataset name
    DATASET = 'hcp'
    
    def __init__(self, subject: str):
        """
        Initialize the FileMapper for a specific subject.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # load HCP subject list
        subject_list = pd.read_csv(os.path.join(DATA_DIRECTORY_HCP, 'subject_list_hcp.csv'))
        # ensure the subject column is string type
        subject_list['subject'] = subject_list['subject'].astype(str)
        # get available subjects in the dataset
        self.available_subjects = subject_list['subject'].unique().astype(str).tolist()
        # check whether any subjects are found
        if not self.available_subjects:
            raise RuntimeError(f"No subjects found in HCP directory: {DATA_DIRECTORY_HCP}")

        # check if subject is valid
        if self.subject not in self.available_subjects:
            raise ValueError(f"Subject '{self.subject}' not found in dataset.")

        # only one session per subject in HCP
        self.sessions = ['01']
        # get the tasks for the subject
        # filter to specific subject
        self.subject_df = subject_list[subject_list['subject'] == self.subject]
        self.tasks = self.subject_df['task'].unique().tolist()
        # if no tasks found, raise error
        if not self.tasks:
            raise RuntimeError(f"No tasks found for subject '{self.subject}' in HCP dataset.")
        # loop through sessions and get runs for each task
        self.tasks_runs = {}
        for task in self.tasks:
            self.tasks_runs[task] = {}
            # filter to specific task
            task_df = self.subject_df[self.subject_df['task'] == task]
            # get runs for the task
            self.tasks_runs[task]['01'] = task_df['acq'].astype(str).tolist()

    def get_fmri_files(
        self, 
        task: str,
        sessions: List[str] | None = None,
        file_type: Literal['prep', 'final'] = 'prep'
    ) -> list[str]:
        """
        Get the fMRI files from all sessions for a specific task in the HCP dataset.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. Not used for HCP, as there is only one session per subject. Kept 
            for compatibility with the FileMapperEuskalibur class.
        file_type : {'prep', 'final'}
            The type of fMRI files to retrieve. 'prep' returns files that
            have been preprocessed with HCP preprocessing pipelines (including ICA-Fix).
            'final' returns files that have undergone additional (final) preprocessing steps.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """

        fmri_files = []
        # check for multiple runs (should have two runs per task in HCP - LR and RL)
        runs = self.tasks_runs[task]['01']
        # if multiple runs, loop through and get files for each run
        for run in runs:
            files = self.get_session_fmri_files('01', task, run=run, desc=file_type)
            fmri_files.extend(files)

        return fmri_files

    def get_physio_files(
        self, 
        task: str,
        sessions: List[str] | None = None,
        return_json: bool = False,
        file_type: Literal['prep', 'final'] = 'prep'
    ) -> list[str] | list[Tuple[str,str]]:
        """
        Get the physiological files from all sessions for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. Not used for HCP, as there is only one session per subject. Kept 
            for compatibility with the FileMapperEuskalibur class.
        return_json : bool
            Whether to return the json sidecar files. There are no json sidecars for
            the HCP physio files, so this parameter is ignored.
        file_type : {'prep', 'final'}
            The type of physio files to retrieve. 'prep' returns raw
            'physio' files from the HCP dataset. 'final' returns
            'final' physio files that have undergone preprocessing.

        Returns
        -------
        list of str
            A list of physiological file paths.
        """
        if file_type == 'prep':
            desc = None
        elif file_type == 'final':
            desc = 'preproc'

        physio_files = []
        # check for multiple runs (should have two runs per task in HCP - LR and RL)
        runs = self.tasks_runs[task]['01']
        for run in runs:
            files = self.get_session_physio_files('01', task, run=run, desc=desc)
            # if return json is True, return tuple with None for json sidecar
            if return_json:
                physio_files.extend([(f, None) for f in files])
            else:
                physio_files.extend(files)

        return physio_files

    def get_event_files(self, task: str, sessions: List[str] | None = None) -> list[list[str]]:
        """
        Get the event files from all sessions for a specific task from the HCP dataset.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. Not used for HCP, as there is only one session per subject. Kept 
            for compatibility with the FileMapperEuskalibur class.

        Returns
        -------
        list of list of str
            A list of event file paths by run.
        """

        event_files = []
        # check for multiple runs (should have two runs per task in HCP - LR and RL)
        runs = self.tasks_runs[task]['01']
        for run in runs:
            run_files = self.get_session_event_files('01', task, run=run)
            event_files.extend(run_files)

        return event_files

    def get_matching_files(
        self,
        file_entities: dict[str, str], 
        file_modality: Literal['physio','fmri'],
        file_type: Literal['prep', 'final'] = 'prep'
    ) -> list[str]:
        """
        Get physio, fmri, or event files matching specific BIDS entities.

        Parameters
        ----------
        file_entities : dict of str to str
            A dictionary specifying the run and task to match. Should have the following keys:
            - 'task': the task identifier
            - 'run': the run identifier (e.g., 'LR' or 'RL')
        file_modality : {'physio', 'fmri'}
            The type of files to retrieve. Options are 'physio' for physiological
            files, 'fmri' for fMRI files.
        file_type : {'prep', 'final'}
            The stage of processing to retrieve. Options are 'prep' for 'raw' files,
            'final' for final files.

        Returns
        -------
        list of str
            A list of matching file paths. Should only return one file path. Returns
            list for compatibility with FileMapperEuskalibur class.
        """
        # raise error if required entities are not provided
        if 'task' not in file_entities or 'run' not in file_entities:
            raise ValueError("file_entities must include 'task' and 'run' keys for HCP dataset.")
        # raise if unrecognized file_type
        if file_type not in ['prep', 'final']:
            raise ValueError("file_type must be 'prep' or 'final'.")
        # determine suffix and extension based on modality
        if file_modality == 'physio':
            if file_type == 'prep':
                out_file = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{file_entities['task']}_{file_entities['run']}_Physio_log.txt"
                )
            elif file_type == 'final':
                out_file = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{file_entities['task']}_{file_entities['run']}_physio_preproc.tsv.gz"
                )

        elif file_modality == 'fmri':
            if file_type == 'prep':
                out_file = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{file_entities['task']}_{file_entities['run']}.nii.gz"
                )
            elif file_type == 'final':
                out_file = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{file_entities['task']}_{file_entities['run']}_preproc.nii.gz"
                )
        else:
            raise ValueError("file_modality must be 'physio' or 'fmri'")

        return [out_file]

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

    def get_session_fmri_files(
        self, 
        session: str, 
        task: str, 
        run: str | None = None,
        desc:  Literal['prep', 'final'] = 'prep'
    ) -> list[str]:
        """
        Get the fMRI files for a specific session and task in the HCP dataset.

        Parameters
        ----------
        session : str
            The session identifier. This is not used for HCP, as there is only one session per subject. Kept 
            for compatibility with the FileMapperEuskalibur class.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned. 
            Should be 'LR' or 'RL' for HCP.
        desc : Literal['prep', 'final']
            The preprocessing stage to filter files. Defaults to 'prep' for
            the output of HCP preprocessing pipelines (including ICA-Fix). Use 'final' for
            files that have undergone additional (final) preprocessing steps.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        if run is None:
            if desc == 'prep':
                fp_lr = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{task}_LR.nii.gz"
                )
                fp_rl = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{task}_RL.nii.gz"
                )
            elif desc == 'final':
                fp_lr = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{task}_LR_preproc.nii.gz"
                )
                fp_rl = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{task}_RL_preproc.nii.gz"
                )
            filenames = [fp_lr, fp_rl]
        else:
            if desc == 'prep':
                fp = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{task}_{run}.nii.gz"
                )
            elif desc == 'final':
                fp = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"func/tfMRI_{self.subject}_{task}_{run}_preproc.nii.gz"
                )
            filenames = [fp]

        return filenames

    def get_session_physio_files(
        self, 
        session: str, 
        task: str, 
        run: str | None = None,
        desc: Literal['preproc'] | None = None
    ) -> list[str]:
        """
        Get the physiological file paths for a specific session and task in the HCP dataset.

        Parameters
        ----------
        session : str
            The session identifier. This is not used for HCP, as there is only one session per subject. 
            Kept for compatibility.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        desc : str, optional
            The preprocessing stage to filter files. Can provide None to get
            'raw' physio files output from HCP dataset. Can provide
            'preproc' to get physio files that have undergone preprocessing.

        Returns
        -------
        list of str
            A list of physiological file paths.
        """
        if run is None:
            if desc is None:
                fp_lr = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{task}_LR_Physio_log.txt"
                )
                fp_rl = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{task}_RL_Physio_log.txt"
                )
            elif desc == 'preproc':
                # there was a typo in the HCP physio filenames where 'preproc' is misspelled as 'prepoc'
                fp_lr = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{task}_LR_Physio_log_physio_prepoc.tsv.gz"
                )
                fp_rl = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{task}_RL_Physio_log_physio_prepoc.tsv.gz"
                )
            filenames = [fp_lr, fp_rl]
        else:
            if desc is None:
                fp = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{task}_{run}_Physio_log.txt"
                )
            elif desc == 'preproc':
                # there was a typo in the HCP physio filenames where 'preproc' is misspelled as 'prepoc'
                fp = os.path.join(
                    DATA_DIRECTORY_HCP,
                    f"physio/tfMRI_{self.subject}_{task}_{run}_Physio_log_physio_prepoc.tsv.gz"
                )
            filenames = [fp]
        return filenames

    def get_session_event_files(
        self, 
        session: str, 
        task: str, 
        run: str | None = None
    ) -> list[str]:
        """
        Get the event (EV) files for a specific session and task from the HCP dataset.

        Parameters
        ----------
        session : str
            The session identifier. This is not used for HCP, as there is only one session per subject. 
            Kept for compatibility.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.

        Returns
        -------
        list of str
            A list of EV files paths.
        """
        # create glob pattern to find EV files
        if run is None:
            pattern = os.path.join(
                DATA_DIRECTORY_HCP,
                f"event/{self.subject}_{task}*.txt"
            )
        else:
            pattern = os.path.join(
                DATA_DIRECTORY_HCP,
                f"event/{self.subject}_{task}_{run}_*.txt"
            )
        return glob(pattern)


def get_dataset_subjects(dataset: str) -> List[str] | dict[str, List[str]]:
    """
    Get the list of available subjects for a specific dataset.

    Parameters
    ----------
    dataset : str
        The dataset name. Options are 'euskalibur' or 'hcp'.

    Returns
    -------
    list of str | dict of str to list of str
        A list of subject identifiers (EUSKALIBUR) or a dictionary mapping tasks to subject lists (HCP).
    """
    if dataset == 'euskalibur':
        # The BIDSLayout initialization can be slow, especially for large datasets
        with warnings.catch_warnings():
            # suppress warnings about soon-to-be-deprecated ignore parameter
            warnings.simplefilter("ignore")
            if IS_DERIVED:
                layout = BIDSLayout(DATA_DIRECTORY_EUSKALIBUR, is_derivative=True)
            else:
                layout = BIDSLayout(DATA_DIRECTORY_EUSKALIBUR, derivatives=True)
        subjects = layout.get_subjects()
    elif dataset == 'hcp':
        subject_list = pd.read_csv(os.path.join(DATA_DIRECTORY_HCP, 'subject_list_hcp.csv'))
        # some subjects do not have data for all tasks, so we will return a dictionary
        tasks = subject_list['task'].unique().tolist()
        subjects = {}
        for task in tasks:
            task_subjects = subject_list[subject_list['task'] == task]['subject'].unique().astype(str).tolist()
            subjects[task] = task_subjects
    else:
        raise ValueError("Dataset must be 'euskalibur' or 'hcp'.")

    return subjects