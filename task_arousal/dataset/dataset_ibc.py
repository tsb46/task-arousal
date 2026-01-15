"""
Class for managing and loading preprocessed dataset files for 
a given subject in the IBC dataset.
"""
import json

from typing import List

import pandas as pd
import nibabel as nib
import numpy as np

from task_arousal.constants import MASK_IBC
from task_arousal.io.file import FileMapper
from .dataset_utils import (
    load_physio as _load_physio,
    load_fmri as _load_fmri,
    to_4d as _to_4d,
    DatasetLoad,
)
from task_arousal.constants import DUMMY_VOLUMES

# Conditions for IBC tasks
IBC_CONDITIONS = json.load(open('task_arousal/dataset/ibc_conditions.json', 'r'))


class DatasetIbc:
    """
    Class for managing and loading preprocessed dataset files for a given subject in 
    the IBC dataset.
    """

    def __init__(self, subject: str):
        """
        Initialize the Dataset class for a specific subject in the IBC dataset.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # map file paths associated to subject
        self.file_mapper = FileMapper(dataset = 'ibc', subject=subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks
        # get available sessions from mapper
        self.sessions = self.file_mapper.sessions
        # attach mask to instance
        self.mask = nib.nifti1.load(MASK_IBC)
    

    def get_tr(self, task) -> float:
        """
        Get the repetition time (TR) for the specified task in the IBC dataset.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        float
            The TR in seconds.
        """
        return self.file_mapper.layout.get_tr(derivatives=True, task=task)

    def load_data(
        self,
        task: str,
        sessions: str | List[str] | None = None,
        concatenate: bool = False,
        normalize: bool = True,
        convert_to_2d: bool = True,
        load_func: bool = True,
        load_physio: bool = False,
        verbose: bool = True
    ) -> DatasetLoad:
        """
        Load the preprocessed dataset files for the subject.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : str or List[str], optional
            The session identifier(s). If None, all sessions will be loaded.
        concatenate : bool, optional
            Whether to concatenate data across sessions. 
            If convert_to_2d is False, this will be ignored. Note, that 
            event data will not be concatenated to preserve trial timing 
            across runs. Default is False.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension. 
            If convert_to_2d is False, this will be ignored. Default is True.
        convert_to_2d : bool, optional
            Whether to convert fMRI data to 2D array (voxels x time points). Default is True.
        load_func : bool, optional
            Whether to load fMRI data. Since this dataset only contains fMRI data, 
            this parameter is kept for API consistency. Parameter is ignored. Default is True.
        load_physio : bool, optional
            No physiological data. Kept for API consistency. Parameter is ignored. Default is False.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        # if task is not None, ensure it's an available task
        if task not in self.tasks:
            raise ValueError(
                f"Task '{task}' is not available for subject '{self.subject}'."
            )
        # get TR for task from BIDS layout - scan metadata is same for all runs of a task 
        task_tr = self.get_tr(task)
        # calculate the amount of time to drop based on DUMMY_VOLUMES and TR
        task_time_to_drop = DUMMY_VOLUMES * task_tr
        # get conditions for task
        try:
            conditions = IBC_CONDITIONS[task]
            # get grouping of conditions, if any
            condition_grouper = conditions.get('condition_grouper', None)
            if condition_grouper is None:
                raise ValueError(f"No condition_grouper found for task '{task}'.")
            # get conditions to ignore, if any
            condition_ignore = conditions.get('condition_ignore', [])
            has_events = True
        except KeyError:
            print(f"Warning: No conditions found for task '{task}'. Events will not be loaded.")
            conditions = []
            has_events = False
            condition_grouper = None
            condition_ignore = []

        # get runs available for task
        runs = self.file_mapper.tasks_runs[task]

        # get sessions avaialble for task
        task_sessions = self.file_mapper.get_sessions_task(task)
        # if sessions are provided, ensure it's a list
        if sessions is not None and not isinstance(sessions, list):
            sessions = [sessions]
        # if sessions are provided, ensure all sessions are available
        if sessions is not None:
            for sess in sessions:
                if sess not in task_sessions:
                    raise ValueError(
                        f"Session '{sess}' is not available for subject '{self.subject}'."
                    )
        else:
            sessions = task_sessions

        if verbose:
            print(f"Loading data for subject '{self.subject}', task '{task}', sessions: {sessions}")
        # initialize dataset dictionary
        dataset = {
            'fmri': [],
            'events': []
        }
        # load files for each session
        for session in sessions:
            if verbose:
                print(f"  Loading session '{session}'...")
            # there should be two phase encoding directions per session: 'ap' and 'pa'
            # loop through phase encodings for session
            for ped in ['pa', 'ap']:
                if verbose and ped is not None:
                    print(f"    Loading phase encoding direction '{ped}'...")

                # if runs is empty, create list with None to loop through at least once
                if len(runs[session]) == 0:
                    runs_session = [None]
                else:
                    runs_session = runs[session]
                # loop through runs for session
                for run in runs_session:
                    if verbose and run is not None:
                        print(f"    Loading run '{run}'...")
                    fmri_files = self.file_mapper.get_session_fmri_files(
                        session, task, ped=ped, run=run, desc='preprocfinal'
                    )
                    # if no fMRI file is found, raise error
                    if len(fmri_files) == 0:
                        # in some scenarios, fmri may be missing or artifacts, skip loading
                        if verbose:
                            print(
                                f"No fMRI file found for session '{session}' and task '{task}' "
                                f"and phase encoding direction '{ped}'."
                            )
                        continue
                    elif len(fmri_files) > 1:
                        # raise error if multiple fMRI files are found
                        raise ValueError(
                            f"Multiple fMRI files found for session '{session}' and task '{task}' and "
                            f"phase encoding direction '{ped}'."
                        )
                    # load fMRI file into 2D array or 4D image
                    fmri_data = self.load_fmri(
                        fmri_files[0], normalize=normalize, convert_to_2d=convert_to_2d, 
                        verbose=verbose
                    )
                    # append data to dataset
                    dataset['fmri'].append(fmri_data)
                    # load event files
                    if has_events:
                        event_file = self.file_mapper.get_session_event_files(
                            session, task, ped=ped, run=run
                        )
                        if len(event_file) == 0:
                            print(
                                f"Warning: No event file found for session '{session}' "
                                f"and task '{task}'."
                            )
                            continue
                        elif len(event_file) > 1:
                            raise ValueError(
                                f"Multiple event files found for session '{session}' "
                                f"and task '{task}'."
                            )
                        # convert event file to dataframe
                        event_df = self.events_to_df(
                            fp_event=event_file[0], # type: ignore
                            ped=ped,
                            session=session,
                            drop_time=task_time_to_drop,
                            condition_grouper=condition_grouper,
                            condition_ignore=condition_ignore
                        )
                        dataset['events'].append(event_df)

        # concatenate data across sessions if requested
        if concatenate:
            if verbose:
                print("Concatenating data across sessions...")
            
            # do not concatenate fmri data if not converted to 2d
            if convert_to_2d:
                dataset['fmri'] = [np.concatenate(dataset['fmri'], axis=0)]
            else:
                print("Warning: fmri data not concatenated because convert_to_2d is False.")
            # events are not concatenated to preserve trial timing across runs

        if verbose:
            print(f"Data loading complete for subject '{self.subject}', task '{task}'.")
        return DatasetLoad(**dataset)

    def events_to_df(
        self, 
        fp_event: str,
        ped: str,
        session: str,
        drop_time: float | None = None,
        condition_grouper: dict | None = None,
        condition_ignore: List[str] | None = None
    ) -> pd.DataFrame:
        """
        Convert 1D onset and duration files (from AFNI)
        to a pandas DataFrame.

        Parameters
        ----------
        fp_event : str
            The path to the event file.
        session : str
            The session identifier.
        ped: str
            The phase encoding direction.
        drop_time : float, optional
            Amount of time (in seconds) to drop from the beginning of the run. If provided,
            onsets will be adjusted accordingly. Default is None.
        condition_grouper : dict, optional
            A dictionary mapping original condition names to grouped condition names.
            If provided, conditions will be grouped accordingly. Default is None.
        condition_ignore : List[str], optional
            A list of condition names to ignore. If provided, events with these condition names
            will be removed from the DataFrame. Default is None.

        Returns
        -------
        pd.DataFrame
            The event data as a pandas DataFrame.
        """
        event_df = pd.read_csv(fp_event, delimiter='\t')
        event_df = event_df.sort_values(by='onset')
        # insert session column
        event_df.insert(0, 'session', session)
        # insert ped column
        event_df.insert(1, 'ped', ped)
        # adjust onsets if drop_time is provided
        if drop_time is not None:
            event_df['onset'] = event_df['onset'] - drop_time
            # remove events that occur before time zero
            event_df = event_df[event_df['onset'] >= 0].reset_index(drop=True)
        # if condition_grouper is provided, map conditions
        if condition_grouper is not None:
            event_df['trial_type'] = event_df['trial_type'].replace(condition_grouper)
        # if condition_ignore is provided, remove those conditions
        if condition_ignore is not None:
            event_df = event_df[~event_df['trial_type'].isin(condition_ignore)].reset_index(drop=True)
        return event_df

    def load_fmri(
        self,
        fp: str,
        normalize: bool = False,
        convert_to_2d: bool = True,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Load the preprocessed fMRI data from a NIfTI file, delegating to shared utils.
        Returns time x voxels if convert_to_2d else a 4D NIfTI image.
        """
        return _load_fmri(
            fp, 
            self.mask, # type: ignore
            normalize=normalize, 
            convert_to_2d=convert_to_2d,
            verbose=verbose
        ) # type: ignore

    def to_4d(
        self,
        fmri_data: np.ndarray
    ) -> nib.nifti1.Nifti1Image:
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.
        """
        return _to_4d(fmri_data, self.mask) # type: ignore

