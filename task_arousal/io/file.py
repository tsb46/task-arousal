"""
Class for iterating over subject fMRI and physio data files
"""
from pathlib import Path
from typing import List, Tuple, Literal

from bids import BIDSLayout

from task_arousal.constants import DATA_DIRECTORY


class FileMapper:
    """
    Maps file paths for a specific subject's fMRI and physiological data.
    """
    
    def __init__(self, subject: str):
        """
        Initialize the FileMapper for a specific subject.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        self.layout = BIDSLayout(DATA_DIRECTORY, derivatives=True)
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
            desc=desc
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
            run=run, desc=desc
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

    def modify_file_name(self, file_path: str, entity_update: dict[str, str]) -> str:
        """
        Modify a BIDS file path by updating specific entities.

        Parameters
        ----------
        file_path : str
            The original BIDS file path.
        entity_update : dict of str to str
            A dictionary specifying the entities to update and their new values.

        Returns
        -------
        str
            The modified BIDS file path.
        """
        # Parse the original file path to get its entities
        entities = self.layout.parse_file_entities(file_path)
        # Update the entities with the provided values
        entities.update(entity_update)
        # Construct the new file path using the updated entities
        new_file_path = self.layout.build_path(entities)
        return new_file_path
