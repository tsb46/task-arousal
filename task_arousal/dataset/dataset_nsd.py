"""
Class for managing and loading preprocessed dataset files for
a given subject in the Natural Scenes Dataset (NSD).
"""

from typing import List, Literal

import pandas as pd
import nibabel as nib
import numpy as np

from task_arousal.io.file import FileMapperNSD
from .dataset_utils import (
    load_physio as _load_physio,
    load_fmri as _load_fmri,
    to_img as _to_img,
    DatasetLoad,
)

NSDIMAGERY_CONDITIONS = ["att"]


class DatasetNsd:
    """
    Class for managing and loading preprocessed dataset files for a given subject in
    the NSD dataset.
    """

    def __init__(self, subject: str):
        """
        Initialize the Dataset class for a specific subject in the NSD dataset.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # map file paths associated to subject
        self.file_mapper = FileMapperNSD(subject=subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks
        # get available sessions from mapper
        self.sessions = self.file_mapper.sessions
        # attach mask to instance
        self.mask = self.file_mapper.get_subject_mask()

    def load_data(
        self,
        task: str,
        func_type: Literal["volume", "surface"] = "volume",
        me_type: Literal["optcomb", "t2", "s0"] = "optcomb",
        sessions: str | List[str] | None = None,
        concatenate: bool = False,
        normalize: bool = True,
        bandpass: tuple[float, float] | None = None,
        load_func: bool = True,
        load_physio: bool = True,
        verbose: bool = True,
    ) -> DatasetLoad:
        """
        Load the preprocessed dataset files for the subject.

        Parameters
        ----------
        func_type : Literal["volume", "surface"], optional
            The type of functional data, either "volume" or "surface". "Surface" is not currently supported for NSD dataset. Default is "volume".
        me_type : Literal["optcomb", "t2", "s0"], optional
            The type of multi-echo data to load. Multi-echo data is not available for NSD dataset,
            so this parameter is ignored. Default is "optcomb".
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
            Default is None.
        load_func : bool, optional
            Whether to load fMRI data. Since this dataset only contains fMRI data,
            this parameter is kept for API consistency. Parameter is ignored. Default is True.
        load_physio : bool, optional
            Whether to load physiological data. Default is True.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        if func_type == "surface":
            raise NotImplementedError("Surface data is not available for NSD dataset.")
        # if task is not None, ensure it's an available task
        if task not in self.tasks:
            raise ValueError(
                f"Task '{task}' is not available for subject '{self.subject}'."
            )
        # get parameters for task
        if task == "nsdimagery":
            conditions = ["att"]
            has_events = True
            # NSD imagery doesn't have physio data, skip loading and print warning if requested
            if verbose and load_physio:
                print(
                    f"Physiological data is not available for task '{task}' in NSD dataset. Skipping physio loading."
                )
            load_physio = False
        elif task == "rest":
            conditions = []
            has_events = False
        else:
            raise ValueError(f"Task '{task}' is not recognized.")

        tr = self.file_mapper.get_tr(task)

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
            "physio": [],
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
                # load physio data if requested
                if not load_physio:
                    if verbose:
                        print("Skipping physio data loading...")
                    physio_df = pd.DataFrame()
                else:
                    # load physio file
                    physio_files = self.file_mapper.get_session_physio_files(
                        session, task, run=run, desc="final"
                    )
                    # check if exactly one physio file is found
                    if len(physio_files) == 0:
                        # in some scenarios, physio may not be recorded, skip loading
                        if verbose:
                            print(
                                f"No physiological file found for session '{session}' and task "
                                f"'{task}' and run '{run if run is not None else ''}'."
                            )
                        continue
                    elif len(physio_files) > 1:
                        raise ValueError(
                            f"Multiple physiological files found for session '{session}' "
                            f"and task '{task}' and run '{run if run is not None else ''}'."
                        )
                    # ensure physio file is string
                    assert isinstance(physio_files[0], str), (
                        f"Expected physio file path to be a string, but got {type(physio_files[0])}."
                    )
                    # load physio file into dataframe
                    physio_df = self.load_physio(physio_files[0], normalize=normalize)
                # load fmri data, if requested
                fmri_files = self.file_mapper.get_session_fmri_files(
                    session, task, run=run, desc="final"
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
                    tr=tr,
                    normalize=normalize,
                    bandpass=bandpass,
                    verbose=verbose,
                )
                # append data to dataset
                dataset["fmri"].append(fmri_data)
                dataset["physio"].append(physio_df)
                # load event files
                if has_events:
                    event_files = self.file_mapper.get_session_event_files(
                        session, task, run=run
                    )
                    if len(event_files) == 0:
                        print(
                            f"Warning: No event file found for session '{session}' "
                            f"and task '{task}'."
                        )
                        continue
                    # convert event file to dataframe
                    event_df = self.events_to_df(
                        event_fp=event_files[0],  # type: ignore
                        session=session,
                        task=task,
                    )
                    dataset["events"].append(event_df)

        # concatenate data across sessions if requested
        if concatenate:
            if verbose:
                print("Concatenating data across sessions...")

            # temporally concatenate physio data
            dataset["physio"] = [
                pd.concat(dataset["physio"], axis=0, ignore_index=True)
            ]
            # temporally concatenate fmri data
            dataset["fmri"] = [np.concatenate(dataset["fmri"], axis=0)]

            # events are not concatenated to preserve trial timing across runs

        if verbose:
            print(f"Data loading complete for subject '{self.subject}', task '{task}'.")
        return DatasetLoad(**dataset)

    def events_to_df(
        self,
        event_fp: str,
        task: str,
        session: str,
    ) -> pd.DataFrame:
        """
        The 'preprocessed' design files provided by the NSD authors are in a format difficult to
        use with standard GLM software (e.g. in Nilearn) - sampled at the functional time points with
        1s as onsets(?). To leverage a more standard design file format, we download the raw BIDS design
        files associated with each run in standard format: trial_type, duration, onset, etc. This should
        be fine as long as no volumes were removed from the functional scans in the preprocessing
        (that would need to be accounted for in the timing of the design files).

        Parameters
        ----------
        event_fp : str
            The path to the BIDS event file.
        session : str
            The session identifier.
        task : str
            The task identifier. The only task with events currently is 'nsdimagery'.

        Returns
        -------
        pd.DataFrame
            The event data as a pandas DataFrame.
        """
        # if task is not nsdimagery, raise error since no events are available
        if task != "nsdimagery":
            raise ValueError(
                f"Events are only available for 'nsdimagery' task in NSD dataset, but got '{task}'."
            )
        # load into pandas dataframe
        event_df_orig = pd.read_csv(event_fp, delimiter="\t")
        if event_df_orig.empty:
            raise ValueError(
                f"No event data found in event file '{event_fp}' for task '{task}' in NSD dataset."
            )
        # create a grouper index to set adjacent cue and present trials to the same trial number, since we want to model them as one event
        event_df_orig["trial_group"] = (
            event_df_orig["trial_type"] == "attention_cue"
        ).cumsum()
        # group by trial group and aggregate onsets and durations, taking the first onset and summing durations, and taking the first trial type (which will be 'attention_cue', but we will rename to 'att' in the next step)
        event_df_agg = (
            event_df_orig.groupby("trial_group").agg(
                {
                    "onset": "first",
                    "duration": "sum",
                    "trial_type": "first",
                }
            )
        ).reset_index(drop=True)
        # aggregate trial types 'attention_cue' and 'present' into 'att' for NSD imagery task
        event_df_agg["trial_type"] = event_df_agg["trial_type"].replace(
            {"attention_cue": "att", "present": "att"}
        )

        return event_df_agg

    def load_physio(self, fp: str, normalize: bool = False) -> pd.DataFrame:
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
        return _load_physio(fp, normalize=normalize)

    def load_fmri(
        self,
        fp: str,
        tr: float,
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
            mask_img=self.mask,  # type: ignore
            normalize=normalize,
            bandpass=bandpass,
            tr=tr,
            verbose=verbose,
        )

    def to_img(
        self, fmri_data: np.ndarray, func_type: Literal["volume", "surface"] = "volume"
    ) -> nib.nifti1.Nifti1Image:
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.

        Parameters
        ----------
        fmri_data : np.ndarray
            The fMRI data as a 2D array of shape (time, voxels).
        func_type : Literal["volume", "surface"], optional
            The type of functional data, either "volume" or "surface". "Surface" is not currently supported for NSD dataset.
            Default is "volume".
        """
        if func_type == "surface":
            raise NotImplementedError(
                "Surface data is not available for NSD dataset. This method is not implemented for surface data."
            )
        return _to_img(fmri_data, mask_img=self.mask)  # type: ignore
