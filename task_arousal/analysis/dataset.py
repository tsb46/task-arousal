"""
Class for managing and loading preprocessed dataset files for 
a given subject.
"""
from typing import List, TypedDict

import pandas as pd
import nibabel as nib
import numpy as np

from nilearn.masking import apply_mask, unmask
from scipy.stats import zscore

from task_arousal.constants import MASK
from task_arousal.io.file import FileMapper

# TypedDict for dataset returned from load_data
class DatasetLoad(TypedDict):
    fmri: List[np.ndarray]
    physio: List[pd.DataFrame]
    events: List[pd.DataFrame]

# conditions for pinel task
PINEL_CONDITIONS = [
    'acalc',
    'amot_left',
    'amot_right',
    'asent',
    'chbh',
    'chbv',
    'vcalc',
    'vmot_left',
    'vmot_right',
    'vsent'
]


class Dataset:
    """
    Class for managing and loading preprocessed dataset files for a given subject.
    """

    def __init__(self, subject: str):
        """
        Initialize the Dataset class for a specific subject.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # map file paths associated to subject
        self.file_mapper = FileMapper(subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks
        # get available sessions from mapper
        self.sessions = self.file_mapper.sessions
        # attach mask to instance
        self.mask = MASK

    def load_data(
        self, 
        task: str, 
        sessions: str | List[str] | None = None,
        concatenate: bool = False,
        normalize: bool = True,
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
            Whether to concatenate data across sessions. Default is False.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension. Default is True.
        """
        # if task is not None, ensure it's an available task
        if task not in self.tasks:
            raise ValueError(
                f"Task '{task}' is not available for subject '{self.subject}'."
            )
        # select conditions based on task
        if task == 'pinel':
            conditions = PINEL_CONDITIONS
        else:
            raise NotImplementedError(f"Conditions for task '{task}' are not defined.")
        # if sessions are provided, ensure it's a list
        if sessions is not None and not isinstance(sessions, list):
            sessions = [sessions]
        # if sessions are provided, ensure all sessions are available
        if sessions is not None:
            for sess in sessions:
                if sess not in self.sessions:
                    raise ValueError(
                        f"Session '{sess}' is not available for subject '{self.subject}'."
                    )
        else:
            sessions = self.sessions

        if verbose:
            print(f"Loading data for subject '{self.subject}', task '{task}', sessions: {sessions}")
        # initialize dataset dictionary
        dataset = {
            'fmri': [],
            'physio': [],
            'events': []
        }
        # load files for each session
        for session in sessions:
            if verbose:
                print(f"  Loading session '{session}'...")
            # load physio file
            physio_files = self.file_mapper.get_session_physio_files(session, task, desc='preproc')
            assert len(physio_files) == 1, f"Expected one physio file for session '{session}' and task '{task}', found {len(physio_files)}."
            physio_df = self.load_physio(physio_files[0], normalize=normalize)
            dataset['physio'].append(physio_df)
            # load fMRI file
            fmri_files = self.file_mapper.get_session_fmri_files(session, task, desc='preprocfinal')
            assert len(fmri_files) == 1, f"Expected one fMRI file for session '{session}' and task '{task}', found {len(fmri_files)}."
            fmri_data = self.load_fmri(fmri_files[0], normalize=normalize)
            dataset['fmri'].append(fmri_data)
            # load event files
            event_files = self.file_mapper.get_session_event_files(session, task)
            assert len(conditions) == len(event_files), f"Expected {len(conditions)} event files for session '{session}' and task '{task}', found {len(event_files)}."
            event_df = self.events_to_df(
                fps_onset=[ef[0] for ef in event_files],
                fps_duration=[ef[1] for ef in event_files],
                conditions=conditions,
                session=session
            )
            dataset['events'].append(event_df)

        # concatenate data across sessions if requested
        if concatenate:
            if verbose:
                print("Concatenating data across sessions...")
            dataset['fmri'] = [np.concatenate(dataset['fmri'], axis=0)]
            dataset['physio'] = [pd.concat(dataset['physio'], axis=0, ignore_index=True)]
            dataset['events'] = [pd.concat(dataset['events'], axis=0, ignore_index=True)]

        return DatasetLoad(**dataset)

    def events_to_df(
        self, 
        fps_onset: List[str], 
        fps_duration: List[str],
        conditions: List[str],
        session: str
    ) -> pd.DataFrame:
        """
        Convert 1D onset and duration files (from AFNI)
        to a pandas DataFrame.

        Parameters
        ----------
        fps_onsets : List[str]
            The paths to the onset files.
        fps_durations : List[str]
            The paths to the duration files.
        conditions : List[str]
            The list of conditions to include in the DataFrame.
        session : str
            The session identifier.

        Returns
        -------
        pd.DataFrame
            The event data as a pandas DataFrame.
        """
        # extract event timings
        event_timing = []
        for c in conditions:
            fp_c_onset = _select_condition(fps_onset, c)
            fp_c_duration = _select_condition(fps_duration, c)
            with open(fp_c_onset, 'r') as f:
                onsets = [float(o) for o in f.read().strip().split(' ')]
            with open(fp_c_duration, 'r') as f:
                durations = [float(d) for d in f.read().strip().split(' ')]

            for onset, duration in zip(onsets, durations):
                event_timing.append((c, onset, duration))

        event_df = pd.DataFrame(event_timing, columns=['trial_type', 'onset', 'duration'])
        event_df = event_df.sort_values(by='onset')
        # insert session column
        event_df.insert(0, 'session', session)
        return event_df
    
    def load_physio(
        self, 
        fp: str,
        normalize: bool = False
    ) -> pd.DataFrame:
        """
        Load the preprocessed physio data from a TSV file.

        Parameters
        ----------
        fp : str
            The path to the physio TSV file.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension. Default is False.

        Returns
        -------
        pd.DataFrame
            The physio data as a pandas DataFrame.
        """
        physio_df = pd.read_csv(fp, sep='\t', compression='gzip')
        if normalize:
            # z-score each column
            physio_df = physio_df.apply(zscore, axis=0) # type: ignore
        return physio_df

    def load_fmri(
        self,
        fp: str,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Load the preprocessed fMRI data from a NIfTI file.

        Parameters
        ----------
        fp : str
            The path to the fMRI NIfTI file.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension. Default is False.

        Returns
        -------
        np.ndarray
            The fMRI data as a numpy array with voxels along the columns
            and time points along the rows.
        """
        fmri_img = nib.load(fp) # type: ignore
        fmri_data = apply_mask(fmri_img, self.mask)
        if normalize:
            fmri_data = zscore(fmri_data, axis=0) 
        return fmri_data # type: ignore
    
    def to_4d(
        self,
        fmri_data: np.ndarray
    ) -> nib.Nifti1Image: # type: ignore
        """
        Convert the 2D fMRI data array back to a 4D NIfTI image.

        Parameters
        ----------
        fmri_data : np.ndarray
            The fMRI data as a numpy array with voxels along the columns
            and time points along the rows.

        Returns
        -------
        nib.Nifti1Image
            The fMRI data as a 4D NIfTI image.
        """
        fmri_img_4d = unmask(fmri_data, self.mask)
        return fmri_img_4d # type: ignore



# select condition from duration and onset file paths
def _select_condition(fps: List[str], condition: str) -> str:
    fp_selected = [fp for fp in fps if condition in fp]
    if len(fp_selected) == 0:
        raise ValueError(f"No files found for condition: {condition}")
    elif len(fp_selected) > 1:
        raise ValueError(f"Multiple files found for condition: {condition}")
    return fp_selected[0]