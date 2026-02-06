"""
Class for managing and loading preprocessed dataset files for
a given subject in the Precision Targeting of Association Networks (PAN) dataset.
"""

import json

from typing import List

import pandas as pd
import nibabel as nib
import numpy as np

from task_arousal.constants import MASK_PAN, DUMMY_VOLUMES, TR_PAN
from task_arousal.io.file import FileMapper
from .dataset_utils import (
    load_fmri as _load_fmri,
    to_img as _to_img,
    DatasetLoad,
)

# Conditions for PAN tasks
PAN_CONDITIONS = json.load(open("task_arousal/dataset/pan_conditions.json", "r"))


class DatasetPan:
    """
    Class for managing and loading preprocessed dataset files for a given subject in
    the PAN dataset.
    """

    def __init__(self, subject: str):
        """
        Initialize the Dataset class for a specific subject in the PAN dataset.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # map file paths associated to subject
        self.file_mapper = FileMapper(dataset="pan", subject=subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks
        # get available sessions from mapper
        self.sessions = self.file_mapper.sessions
        # attach mask to instance
        self.mask = nib.nifti1.load(MASK_PAN)

    def load_data(
        self,
        task: str,
        sessions: str | List[str] | None = None,
        concatenate: bool = False,
        normalize: bool = True,
        bandpass: tuple[float, float] | None = None,
        load_func: bool = True,
        load_physio: bool = False,
        verbose: bool = True,
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
            Note, that event data will not be concatenated to preserve trial timing
            across runs. Default is False.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension.
            Default is True.
        bandpass : tuple of float | None, optional
            If provided, apply a Butterworth bandpass filter with these (low, high) frequencies in Hz.
            TR is assumed to be 1.355s for PAN dataset. Default is None.
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
        # get conditions for task
        try:
            conditions = PAN_CONDITIONS[task]
            has_events = True
        except KeyError:
            print(
                f"Warning: No conditions found for task '{task}'. Events will not be loaded."
            )
            conditions = []
            has_events = False

        # print processing info - normalize, bandpass
        if verbose:
            if normalize:
                print("fMRI data will be z-scored per voxel.")
            if bandpass is not None:
                print(
                    f"fMRI data will be bandpass filtered between {bandpass[0]} Hz and {bandpass[1]} Hz."
                )

        # get runs for each session
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
            print(
                f"Loading data for subject '{self.subject}', task '{task}', sessions: {sessions}"
            )
        # initialize dataset dictionary
        dataset = {
            "fmri": [],
            "events": [],
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
                fmri_files = self.file_mapper.get_session_fmri_files(
                    session, task, run=run, desc="preprocfinal"
                )
                # if no fMRI file is found, raise error
                if len(fmri_files) == 0:
                    # in some scenarios, fmri may be missing or artifacts, skip loading
                    if verbose:
                        print(
                            f"No fMRI file found for session '{session}' and task '{task}'."
                        )
                    continue
                elif len(fmri_files) > 1:
                    # raise error if multiple fMRI files are found
                    raise ValueError(
                        f"Multiple fMRI files found for session '{session}' and task '{task}'."
                    )
                # load fMRI file into 2D array or 4D image
                fmri_data = self.load_fmri(
                    fmri_files[0],
                    normalize=normalize,
                    bandpass=bandpass,
                    verbose=verbose,
                )
                # append data to dataset
                dataset["fmri"].append(fmri_data)
                # load event files
                if has_events:
                    onset_files = self.file_mapper.get_session_event_files(
                        session, task, run=run
                    )
                    if len(onset_files) == 0:
                        print(
                            f"Warning: No event file found for session '{session}' "
                            f"and task '{task}'."
                        )
                        continue
                    # convert event file to dataframe
                    event_df = self.events_to_df(
                        fp_onsets=onset_files,  # type: ignore
                        session=session,
                        task=task,
                    )
                    dataset["events"].append(event_df)

        # concatenate data across sessions if requested
        if concatenate:
            if verbose:
                print("Concatenating data across sessions...")

            # temporally concatenate fmri data
            dataset["fmri"] = [np.concatenate(dataset["fmri"], axis=0)]

            # events are not concatenated to preserve trial timing across runs

        if verbose:
            print(f"Data loading complete for subject '{self.subject}', task '{task}'.")
        return DatasetLoad(**dataset)

    def events_to_df(
        self,
        fp_onsets: List[str],
        task: str,
        session: str,
    ) -> pd.DataFrame:
        """
        Convert 1D onset files (from AFNI) to a pandas DataFrame. Durations are
        set in the conditions .json file.

        Parameters
        ----------
        fp_onsets : List[str]
            The paths to the event onset .1D files.
        session : str
            The session identifier.

        Returns
        -------
        pd.DataFrame
            The event data as a pandas DataFrame.
        """
        # loop through conditions of task to build event dataframe
        event_timing = []
        for c in PAN_CONDITIONS[task]["conditions"]:
            c_onsets = _extract_timings_pan(fp_onsets=fp_onsets, task=task, condition=c)
            duration = PAN_CONDITIONS[task]["duration"][c]

            for onset in c_onsets:
                event_timing.append((c, onset, duration))

        event_df = pd.DataFrame(
            event_timing, columns=["trial_type", "onset", "duration"]
        )
        event_df = event_df.sort_values(by="onset")
        # insert session column
        event_df.insert(0, "session", session)
        return event_df

    def load_fmri(
        self,
        fp: str,
        normalize: bool = False,
        bandpass: tuple[float, float] | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Load the preprocessed fMRI data from a NIfTI file, delegating to shared utils.
        Returns time x voxels matrix.
        """
        return _load_fmri(
            fp,
            self.mask,  # type: ignore
            normalize=normalize,
            bandpass=bandpass,
            tr=TR_PAN,
            verbose=verbose,
        )

    def to_img(self, fmri_data: np.ndarray) -> nib.nifti1.Nifti1Image:
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.
        """
        return _to_img(fmri_data, self.mask)  # type: ignore


def _extract_timings_pan(
    fp_onsets: List[str],
    task: str,
    condition: str,
) -> List[float]:
    """
    Extract onset timings for a specific condition from a PAN onset file.

    Parameters
    ----------
    fp_onsets : List[str]
        The paths to the event onset .1D files.
    condition : str
        The condition to extract timings for.

    Returns
    -------
    List[float]
        A list of onset timings for the specified condition.
    """
    # find onset files for condition
    fp_condition_onset = [fp for fp in fp_onsets if f"{condition}.1D" in fp]
    if len(fp_condition_onset) == 0:
        raise ValueError(
            f"No onset files found for condition: {condition} in {task} task"
        )
    elif len(fp_condition_onset) > 1:
        raise ValueError(
            f"Multiple onset files found for condition: {condition} in {task} task"
        )
    # load onset files for condition
    with open(fp_condition_onset[0], "r") as f:
        c_onsets = [float(o) for o in f.read().strip().split(" ")]

    # subtract dummy volume duration from onsets
    c_onsets = [o - (DUMMY_VOLUMES * TR_PAN) for o in c_onsets]
    # remove onsets that are negative after dummy volume removal
    c_onsets = [o for o in c_onsets if o >= 0]
    return c_onsets
