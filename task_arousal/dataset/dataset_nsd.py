"""
Class for managing and loading preprocessed dataset files for
a given subject in the Natural Scenes Dataset (NSD).
"""

from typing import List, Literal

import pandas as pd
import nibabel as nib
import numpy as np

from task_arousal.constants import TR_NSD
from task_arousal.io.file import FileMapperNSD
from .dataset_utils import (
    load_physio as _load_physio,
    load_fmri as _load_fmri,
    to_img as _to_img,
    DatasetLoad,
)


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
            Whether to load physiological data. Default is True.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        if func_type == "surface":
            raise NotImplementedError(
                "Surface data is not available for NSD dataset. This method is not implemented for surface data."
            )
        # if task is not None, ensure it's an available task
        if task not in self.tasks:
            raise ValueError(
                f"Task '{task}' is not available for subject '{self.subject}'."
            )
        # get conditions for task
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
                                f"{task}' and run '{run if run is not None else ''}'."
                            )
                        continue
                    elif len(physio_files) > 1:
                        breakpoint()
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
                    normalize=normalize,
                    bandpass=bandpass,
                    verbose=verbose,
                )
                # append data to dataset
                dataset["fmri"].append(fmri_data)
                dataset["physio"].append(physio_df)
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
        raise NotImplementedError(
            "Event data is not available for NSD dataset. This method is not implemented."
        )

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
            tr=TR_NSD,
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
