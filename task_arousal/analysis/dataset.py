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
    fmri: List[np.ndarray] | List[nib.Nifti1Image] # type: ignore
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

# conditions for simon task
SIMON_CONDITIONS = [
    'left_congruent',
    'right_congruent',
    'left_incongruent',
    'right_incongruent'
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
        convert_to_2d: bool = True,
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
            If convert_to_2d is False, this will be ignored. Default is False.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension. 
            If convert_to_2d is False, this will be ignored. Default is True.
        convert_to_2d : bool, optional
            Whether to convert fMRI data to 2D array (voxels x time points). Default is True.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        # if task is not None, ensure it's an available task
        if task not in self.tasks:
            raise ValueError(
                f"Task '{task}' is not available for subject '{self.subject}'."
            )
        # select conditions and runs based on task
        if task == 'pinel':
            conditions = PINEL_CONDITIONS
            has_events = True
            runs = self.file_mapper.tasks_runs[task]
        elif task == 'simon':
            conditions = SIMON_CONDITIONS
            has_events = True
            runs = self.file_mapper.tasks_runs[task]
        elif task in 'breathhold':
            conditions = []
            has_events = False
            runs = self.file_mapper.tasks_runs[task]
        elif task == 'rest':
            conditions = []
            has_events = False
            runs = self.file_mapper.tasks_runs[task]
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
            
            # if runs is empty, create list with None to loop through at least once
            if len(runs[session]) == 0:
                runs_session = [None]
            else:
                runs_session = runs[session]

            # loop through runs for session
            for run in runs_session:
                if verbose and run is not None:
                    print(f"    Loading run '{run}'...")
                # load physio file
                physio_files = self.file_mapper.get_session_physio_files(session, task, run=run, desc='preproc')
                # check if exactly one physio file is found
                if len(physio_files) == 0:
                    # in some scenarios, physio may not be recorded, skip loading
                    if verbose:
                        print(
                            f"No physiological file found for session '{session}' and task "
                            f"{task}' and run '{run if run is not None else ''}'."
                        )
                    continue
                elif len(physio_files) > 1:
                    raise ValueError(
                        f"Multiple physiological files found for session '{session}' "
                        f"and task '{task}' and run '{run if run is not None else ''}'."
                    )
                # load physio file into dataframe
                physio_df = self.load_physio(physio_files[0], normalize=normalize)
            
                # load fMRI file
                fmri_files = self.file_mapper.get_session_fmri_files(session, task, run=run, desc='preprocfinal')
                # if no fMRI file is found, raise error
                if len(fmri_files) == 0:
                    # in some scenarios, fmri may be missing or artifacts, skip loading
                    if verbose:
                        print(
                            f"No fMRI file found for session '{session}' and task "
                            f"'{task}' and run '{run if run is not None else ''}'."
                        )
                    continue
                elif len(fmri_files) > 1:
                    # raise error if multiple fMRI files are found
                    raise ValueError(
                        f"Multiple fMRI files found for session '{session}' and task '{task}' and "
                        f"run '{run if run is not None else ''}'."
                    )
                # load fMRI file into 2D array or 4D image
                fmri_data = self.load_fmri(fmri_files[0], normalize=normalize, convert_to_2d=convert_to_2d)
                # append data to dataset
                dataset['physio'].append(physio_df)
                dataset['fmri'].append(fmri_data)
                # load event files
                if has_events:
                    event_files = self.file_mapper.get_session_event_files(session, task)
                    event_df = self.events_to_df(
                        fps_onset=[ef[0] for ef in event_files],
                        fps_duration=[ef[1] for ef in event_files],
                        conditions=conditions,
                        task=task,
                        session=session
                    )
                    dataset['events'].append(event_df)

        # concatenate data across sessions if requested
        if concatenate:
            if verbose:
                print("Concatenating data across sessions...")
            
            dataset['physio'] = [pd.concat(dataset['physio'], axis=0, ignore_index=True)]
            # do not concatenate fmri data if not converted to 2d
            if convert_to_2d:
                dataset['fmri'] = [np.concatenate(dataset['fmri'], axis=0)]
            else:
                print("Warning: fmri data not concatenated because convert_to_2d is False.")
            # only concatenate events if they exist
            if has_events:
                dataset['events'] = [pd.concat(dataset['events'], axis=0, ignore_index=True)]

        if verbose:
            print("Data loading complete.")
        return DatasetLoad(**dataset)

    def events_to_df(
        self, 
        fps_onset: List[str], 
        fps_duration: List[str],
        conditions: List[str],
        task: str,
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
            if task == 'pinel':
                onsets, durations = _extract_timing_pinel(
                    fps_onset=fps_onset,
                    fps_duration=fps_duration,
                    condition=c
                )
            elif task == 'simon':
                onsets, durations = _extract_timing_simon(
                    fps_onset=fps_onset, 
                    condition=c
                )
            else:
                raise NotImplementedError(f"Event extraction not implemented for task '{task}'")

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
        normalize: bool = False,
        convert_to_2d: bool = True
    ) -> np.ndarray:
        """
        Load the preprocessed fMRI data from a NIfTI file.

        Parameters
        ----------
        fp : str
            The path to the fMRI NIfTI file.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension. 
            If convert_to_2d is False, this will be ignored. Default is False.
        convert_to_2d : bool, optional
            Whether to convert fMRI data to 2D array (voxels x time points). Default is True.

        Returns
        -------
        np.ndarray
            The fMRI data as a numpy array with voxels along the columns
            and time points along the rows.
        """
        fmri_img = nib.load(fp) # type: ignore
        if not convert_to_2d:
            return fmri_img # type: ignore

        # apply mask to get 2D array (voxels x time points)
        fmri_data = apply_mask(fmri_img, self.mask)
        if normalize:
            fmri_data = zscore(fmri_data, axis=0)
            # check that z-score did not introduce NaNs
            if np.isnan(np.array(fmri_data)).any():
                raise ValueError(f"NaN values found in fMRI data after z-scoring for file: {fp}")
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


def _extract_timing_pinel(
    fps_onset: List[str],
    fps_duration: List[str],
    condition: str
) -> tuple[List[float], List[float]]:
    """Extract timing information for a specific condition from onset and duration file paths for pinel task.

    Parameters:
    ----------
    fps_onset : List[str]
        List of file paths to onset files.
    fps_duration : List[str]
        List of file paths to duration files.
    condition : str
        The condition to extract timing information for.

    Raises:
        ValueError: If no onset files are found for the condition.
        ValueError: If multiple onset files are found for the condition.
        ValueError: If no duration files are found for the condition.
        ValueError: If multiple duration files are found for the condition.

    Returns:
        tuple[List[float], List[float]]: A tuple containing two lists: the onsets and durations for the condition.
    """
    # find onset files for condition
    fp_condition_onset = [fp for fp in fps_onset if condition in fp]
    if len(fp_condition_onset) == 0:
            raise ValueError(f"No onset files found for condition: {condition} in pinel task")
    elif len(fp_condition_onset) > 1:
        raise ValueError(f"Multiple onset files found for condition: {condition} in pinel task")
    # load onset files for condition
    with open(fp_condition_onset[0], 'r') as f:
        c_onsets = [float(o) for o in f.read().strip().split(' ')]

    # find duration files for condition
    fp_condition_duration = [fp for fp in fps_duration if condition in fp]
    if len(fp_condition_duration) == 0:
        raise ValueError(f"No duration files found for condition: {condition} in pinel task")
    elif len(fp_condition_duration) > 1:
        raise ValueError(f"Multiple duration files found for condition: {condition} in pinel task")
    # load duration files for condition
    with open(fp_condition_duration[0], 'r') as f:
        c_durations = [float(d) for d in f.read().strip().split(' ')]

    # extract timing information from file paths for pinel task
    return c_onsets, c_durations


def _extract_timing_simon(
    fps_onset: List[str],
    condition: str
) -> tuple[List[float], List[float]]:
    """
    Extract timing information for a specific condition from onset and duration
    file paths for simon task. The simon task has onset and duration in the onset
    files, so only the onset files are used. Each space-separated value has the onset
    in the first position, and the duration in the second position, separated by a colon ':'.
    In some cases, the subject responded correctly to all trials of a condition. In these cases,
    the onset file will contain one entry with a value of '-1:0' to indicate no incorrect trials.

    Parameters:
    ----------
    fps_onset : List[str]
        List of file paths to onset files.
    condition : str
        The condition to extract timing information for.

    Raises:
        ValueError: If no onset files are found for the condition.
        ValueError: If more than two onset files are found for the condition (correct and incorrect conditions).

    Returns:
        tuple[List[float], List[float]]: A tuple containing two lists: the onsets and durations for the condition.
    """
    # find onset files for condition
    fps_condition_onset = [fp for fp in fps_onset if condition in fp]
    if len(fps_condition_onset) == 0:
            raise ValueError(f"No onset files found for condition: {condition} in simon task")
    elif len(fps_condition_onset) == 1:
        raise ValueError(f"Only one onset file found for condition: {condition} in simon task. Two files expected (correct and incorrect conditions).")
    elif len(fps_condition_onset) > 2:
        raise ValueError(f"More than two onset files found for condition: {condition} in simon task")
    
    # load onset files for correct and incorrect conditions
    c_onsets = []
    c_durations = []
    for fp in fps_condition_onset:
        with open(fp, 'r') as f:
            c_onset_duration = [o.split(':') for o in f.read().strip().split(' ')]
            try:
                c_onsets.extend([float(od[0]) for od in c_onset_duration if float(od[0]) >= 0])
                c_durations.extend([float(od[1]) for od in c_onset_duration if float(od[0]) >= 0])
            except IndexError:
                raise ValueError(
                    f"Onset file for condition: {condition} in simon task does not have correct format."
                    " Each value should have onset and duration separated by a colon ':'."
                )

    # extract timing information from file paths for simon task
    return c_onsets, c_durations
