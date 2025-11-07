"""
Subject-level dataset loader for HCP preprocessed data.
"""
from __future__ import annotations

from typing import List, Dict

import pandas as pd
import numpy as np
import nibabel as nib

from task_arousal.constants import MASK_HCP
from task_arousal.io.file import FileMapperHCP
from .dataset_utils import (
    load_physio as _load_physio, 
    load_fmri as _load_fmri,
    to_4d as _to_4d,
    DatasetLoad,
)


# expected task columns
TASK_HCP_EV_COLUMNS = ['onset', 'duration', 'amplitude']

# conditions for emotion task in HCP dataset
EMOTION_CONDITIONS = [
    'fear',
    'neut'
]
# conditions for gambling task in HCP dataset
GAMBLING_CONDITIONS = [
    'loss',
    'win'
]
# conditions for language task in HCP dataset
LANGUAGE_CONDITIONS = [
    'math',
    'story' 
]
# conditions for motor task in HCP dataset
MOTOR_CONDITIONS = [
    'cue',
    'lf',
    'rf',
    'rh',
    'lh',
    't'
]
# conditions for relational task in HCP dataset
RELATIONAL_CONDITIONS = [
    'relation',
    'match'
]
# conditions for social task in HCP dataset
SOCIAL_CONDITIONS = [
    'mental',
    'theory_of_mind'
]
# conditions for working memory task in HCP dataset
WM_CONDITIONS = [
    '2bk_body',
    '2bk_face',
    '2bk_place',
    '2bk_tool',
    '0bk_body',
    '0bk_face',
    '0bk_place',
    '0bk_tool'  
]


class DatasetHCPSubject:
    """
    Load preprocessed HCP data for a subject across runs of a task.
    """

    def __init__(self, subject: str):
        self.subject = str(subject)
        self.file_mapper = FileMapperHCP(self.subject)
        self.tasks = self.file_mapper.tasks
        self.sessions = self.file_mapper.sessions  # ['01']
        self.mask = MASK_HCP

    def load_data(
        self,
        task: str,
        sessions: str | List[str] | None = None,
        concatenate: bool = False,
        normalize: bool = True,
        convert_to_2d: bool = True,
        load_func: bool = True,
        load_physio: bool = True,
        verbose: bool = True
    ) -> DatasetLoad:
        """
        Load HCP subject data for a given task across all runs (LR/RL).

        Parameters
        ----------
        task : str
            Task name to load (e.g., 'EMOTION', 'GAMBLING').
        sessions : str | List[str] | None
            Session(s) to load. This parameter is unused for HCP as there is only one session ('01'). 
            Kept for API consistency.
        concatenate : bool
            If True, concatenate data across runs. If convert_to_2d is False, 
            this will be ignored. Note, that event data will not be concatenated 
            to preserve trial timing across runs. Default is False.
        normalize : bool
            If True, z-score normalize the data.
        convert_to_2d : bool
            If True, convert fMRI data to 2D array (time x voxels).
        load_func : bool
            If True, load fMRI data.
        load_physio : bool
            If True, load physiological data.
        verbose : bool
            If True, print progress messages.
        """
        if task not in self.tasks:
            raise ValueError(f"Task '{task}' is not available for subject '{self.subject}'.")

        # HCP: one session ('01'), with runs per task (e.g., LR, RL)
        session = '01'
        runs = self.file_mapper.tasks_runs[task][session]
        if verbose:
            print(f"Loading HCP data for subject '{self.subject}', task '{task}', runs: {runs}")

        dataset: Dict[str, list] = {"fmri": [], "physio": [], "events": []}

        for run in runs:
            if verbose:
                print(f"  Loading run '{run}'...")

            # physio data
            if not load_physio:
                if verbose:
                    print("    Skipping physio data loading...")
                physio_df = pd.DataFrame()
            else:
                physio_files = self.file_mapper.get_session_physio_files(session, task, run=run, desc='preproc')
                if len(physio_files) == 0:
                    if verbose:
                        print(f"    No physio file found for run '{run}'.")
                    physio_df = pd.DataFrame()
                elif len(physio_files) > 1:
                    raise ValueError(f"Multiple physio files found for run '{run}'.")
                else:
                    physio_df = _load_physio(physio_files[0], normalize=normalize)

            # fMRI data
            if not load_func:
                if verbose:
                    print("    Skipping fMRI data loading...")
                fmri_data = np.array([])
            else:
                fmri_files = self.file_mapper.get_session_fmri_files(session, task, run=run, desc='final')
                if len(fmri_files) == 0:
                    if verbose:
                        print(f"    No fMRI file found for run '{run}'.")
                    fmri_data = np.array([])
                elif len(fmri_files) > 1:
                    raise ValueError(f"Multiple fMRI files found for run '{run}'.")
                else:
                    fmri_data = _load_fmri(
                        fmri_files[0], 
                        self.mask, 
                        normalize=normalize, 
                        convert_to_2d=convert_to_2d,
                        verbose=verbose
                    )

            # events
            ev_files = self.file_mapper.get_session_event_files(session, task, run=run)
            events_df = self._events_to_df(ev_files, task=task, session=session, run=run)

            dataset['physio'].append(physio_df)
            dataset['fmri'].append(fmri_data)
            dataset['events'].append(events_df)

        if concatenate:
            if verbose:
                print("Concatenating data across runs...")
            # physio
            if len(dataset['physio']) and not all(df.empty for df in dataset['physio']):
                dataset['physio'] = [pd.concat(dataset['physio'], axis=0, ignore_index=True)]
            else:
                dataset['physio'] = [pd.DataFrame()]
            # fmri
            if convert_to_2d and len(dataset['fmri']) and all(isinstance(x, np.ndarray) and x.size > 0 for x in dataset['fmri']):
                dataset['fmri'] = [np.concatenate(dataset['fmri'], axis=0)]
            elif not convert_to_2d:
                print("Warning: fmri data not concatenated because convert_to_2d is False.")
            # events are not concatenated to preserve trial timing across runs

        if verbose:
            print(f"HCP data loading complete for subject '{self.subject}', task '{task}'.")
        return DatasetLoad(**dataset)  # type: ignore

    def _events_to_df(self, ev_files: List[str], task: str, session: str, run: str) -> pd.DataFrame:
        """
        Parse HCP EV files into a tidy DataFrame with columns:
        ['session', 'run', 'trial_type', 'onset', 'duration', 'amplitude'].
        """
        # get task conditions based on task
        if task == 'EMOTION':
            conditions = EMOTION_CONDITIONS
        elif task == 'GAMBLING':
            conditions = GAMBLING_CONDITIONS
        elif task == 'LANGUAGE':    
            conditions = LANGUAGE_CONDITIONS
        elif task == 'MOTOR':
            conditions = MOTOR_CONDITIONS
        elif task == 'RELATIONAL':
            conditions = RELATIONAL_CONDITIONS
        elif task == 'SOCIAL':
            conditions = SOCIAL_CONDITIONS
        elif task == 'WM':
            conditions = WM_CONDITIONS
        else:
            raise ValueError(f"Unknown task '{task}' for event parsing.")
        
        # loop through conditions and find corresponding files
        events_df = []
        for cond in conditions:
            matching_files = [f for f in ev_files if ((f"RL_{cond}.txt" in f) or (f"LR_{cond}.txt" in f))]
            if len(matching_files) > 1:
                raise ValueError(f"Multiple event files found for condition '{cond}' in run '{run}'.")
            if len(matching_files) < 1:
                raise ValueError(f"No event file found for condition '{cond}' in run '{run}'.")
            elif len(matching_files) == 1:
                ev_file = matching_files[0]

            # load the event file
            cond_data = np.loadtxt(ev_file)  # shape: (n_events, 3)
            if cond_data.ndim == 1:
                cond_data = cond_data[np.newaxis, :]  # ensure 2D
            # convert to DataFrame
            cond_df = pd.DataFrame(cond_data, columns=TASK_HCP_EV_COLUMNS)
            # insert trial_type, session, run
            cond_df.insert(0, 'trial_type', cond)
            cond_df.insert(0, 'run', run)
            cond_df.insert(0, 'session', session)
            events_df.append(cond_df)

        # concatenate all conditions
        events_df = pd.concat(events_df, axis=0, ignore_index=True)
        # sort by onset time
        events_df = events_df.sort_values(by='onset').reset_index(drop=True)
        return events_df

    def to_4d(
        self,
        fmri_data: np.ndarray
    ) -> nib.nifti1.Nifti1Image: 
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.
        """
        # get mask
        mask_img = nib.nifti1.load(MASK_HCP)
        # ensure nifti
        assert isinstance(mask_img, nib.nifti1.Nifti1Image), "Loaded mask is not a Nifti1Image."
        return _to_4d(fmri_data, mask_img)
