"""
Class for managing and loading preprocessed dataset files for
a given subject in the Euskalibur dataset.
"""

from typing import List, Literal

import pandas as pd
import nibabel as nib
import numpy as np

from task_arousal.constants import MASK_EUSKALIBUR
from task_arousal.io.file import FileMapper
from .dataset_utils import (
    load_physio as _load_physio,
    load_fmri as _load_fmri,
    to_img as _to_img,
    DatasetLoad,
)

# conditions for pinel task in Euskalibur dataset
PINEL_CONDITIONS = [
    "acalc",
    "amot_left",
    "amot_right",
    "asent",
    "chbh",
    "chbv",
    "vcalc",
    "vmot_left",
    "vmot_right",
    "vsent",
]

# conditions for simon task in Euskalibur dataset
SIMON_CONDITIONS = [
    "left_congruent",
    "right_congruent",
    "left_incongruent",
    "right_incongruent",
]

# conditions for motor task in Euskalibur dataset
MOTOR_CONDITIONS = [
    "toe_left",
    "toe_right",
    "finger_left",
    "finger_right",
    "tongue",
    "star",
]


class DatasetEuskalibur:
    """
    Class for managing and loading preprocessed dataset files for a given subject in
    the Euskalibur dataset.
    """

    def __init__(self, subject: str):
        """
        Initialize the Dataset class for a specific subject in the Euskalibur dataset.

        Parameters
        ----------
        subject : str
            The subject identifier.
        """
        self.subject = subject
        # map file paths associated to subject
        self.file_mapper = FileMapper(dataset="euskalibur", subject=subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks
        # get available sessions from mapper
        self.sessions = self.file_mapper.sessions
        # attach mask to instance
        self.mask = nib.nifti1.load(MASK_EUSKALIBUR)
        # ensure mask is Nifti1Image
        assert isinstance(self.mask, nib.nifti1.Nifti1Image), (
            "Mask is not a Nifti1Image."
        )
        # initialize surface template attribute for later use in surface data loading and to_img conversion
        self.surface_template = None

    def load_data(
        self,
        task: str,
        func_type: Literal["volume", "surface"] = "volume",
        sessions: str | List[str] | None = None,
        concatenate: bool = False,
        normalize: bool = True,
        load_func: bool = True,
        load_physio: bool = True,
        verbose: bool = True,
    ) -> DatasetLoad:
        """
        Load the preprocessed dataset files for the subject.

        Parameters
        ----------
        task : str
            The task identifier.
        func_type : {'volume', 'surface'}, optional
            The type of functional data to load. Default is 'volume'.
        sessions : str or List[str], optional
            The session identifier(s). If None, all sessions will be loaded.
        concatenate : bool, optional
            Whether to concatenate data across sessions.
            Note, that event data will not be concatenated to preserve trial timing
            across runs. Default is False.
        normalize : bool, optional
            Whether to normalize (z-score) the data along the time dimension.
            Default is True.
        load_func : bool, optional
            Whether to load fMRI data. Default is True.
        load_physio : bool, optional
            Whether to load physiological data. Default is True.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        # if task is not None, ensure it's an available task
        if task not in self.tasks:
            raise ValueError(
                f"Task '{task}' is not available for subject '{self.subject}'."
            )
        # select conditions and runs based on task
        if task == "pinel":
            conditions = PINEL_CONDITIONS
            has_events = True
            runs = self.file_mapper.tasks_runs[task]
        elif task == "simon":
            conditions = SIMON_CONDITIONS
            has_events = True
            runs = self.file_mapper.tasks_runs[task]
        elif task == "motor":
            conditions = MOTOR_CONDITIONS
            has_events = True
            runs = self.file_mapper.tasks_runs[task]
        elif task in "breathhold":
            conditions = []
            has_events = False
            runs = self.file_mapper.tasks_runs[task]
        elif task == "rest":
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
            print(
                f"Loading data for subject '{self.subject}', task '{task}', sessions: {sessions}"
            )
        # initialize dataset dictionary
        dataset = {"fmri": [], "physio": [], "events": []}
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
                        session, task, run=run, desc="preproc"
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
                        raise ValueError(
                            f"Multiple physiological files found for session '{session}' "
                            f"and task '{task}' and run '{run if run is not None else ''}'."
                        )
                    # load physio file into dataframe
                    physio_df = self.load_physio(physio_files[0], normalize=normalize)

                # load fMRI file
                if not load_func:
                    if verbose:
                        print("Skipping fMRI data loading...")
                    fmri_data = np.array([])
                else:
                    fmri_files = self.file_mapper.get_session_fmri_files(
                        session,
                        task,
                        run=run,
                        desc="preprocfinal",
                        extension=".nii.gz"
                        if func_type == "volume"
                        else ".dtseries.nii",
                    )
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
                    # If we are in surface mode, store a template dtseries path for later to_img()
                    if func_type == "surface" and self.surface_template is None:
                        self.surface_template = fmri_files[0]
                        if verbose:
                            print(
                                f"Using surface template dtseries: {self.surface_template}"
                            )
                    # load fMRI file into 2D array or 4D image
                    fmri_data = self.load_fmri(
                        fmri_files[0],
                        func_type=func_type,
                        normalize=normalize,
                        verbose=verbose,
                    )

                # append data to dataset
                dataset["physio"].append(physio_df)
                dataset["fmri"].append(fmri_data)
                # load event files
                if has_events:
                    event_files = self.file_mapper.get_session_event_files(
                        session, task
                    )
                    event_df = self.events_to_df(
                        fps_onset=[ef[0] for ef in event_files],
                        fps_duration=[ef[1] for ef in event_files],
                        conditions=conditions,
                        task=task,
                        session=session,
                    )
                    dataset["events"].append(event_df)

        # concatenate data across sessions if requested
        if concatenate:
            if verbose:
                print("Concatenating data across sessions...")

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
        fps_onset: List[str],
        fps_duration: List[str],
        conditions: List[str],
        task: str,
        session: str,
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
            if task == "pinel" or task == "motor":
                onsets, durations = _extract_timing_pinel_motor(
                    fps_onset=fps_onset, fps_duration=fps_duration, condition=c
                )
            elif task == "simon":
                onsets, durations = _extract_timing_simon(
                    fps_onset=fps_onset, condition=c
                )
            else:
                raise NotImplementedError(
                    f"Event extraction not implemented for task '{task}'"
                )

            for onset, duration in zip(onsets, durations):
                event_timing.append((c, onset, duration))

        event_df = pd.DataFrame(
            event_timing, columns=["trial_type", "onset", "duration"]
        )
        event_df = event_df.sort_values(by="onset")
        # insert session column
        event_df.insert(0, "session", session)
        return event_df

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
        func_type: Literal["volume", "surface"] = "volume",
        normalize: bool = False,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Load the preprocessed fMRI data from a NIfTI file, delegating to shared utils.
        Returns time x voxels matrix
        """
        return _load_fmri(
            fp,
            func_type=func_type,
            mask_img=self.mask,  # type: ignore
            normalize=normalize,
            verbose=verbose,
        )

    def to_img(
        self,
        fmri_data: np.ndarray,
        func_type: Literal["volume", "surface"] = "volume",
        surface_template: str | nib.cifti2.cifti2.Cifti2Image | None = None,
    ) -> nib.nifti1.Nifti1Image | nib.cifti2.cifti2.Cifti2Image:
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.
        """
        if func_type == "surface":
            template = (
                surface_template
                if surface_template is not None
                else self.surface_template
            )
            if template is None:
                raise ValueError(
                    "surface_template is required for func_type='surface'. "
                    "Call load_data(..., func_type='surface') first (to auto-set it), "
                    "or pass surface_template explicitly."
                )
            return _to_img(fmri_data, func_type=func_type, surface_template=template)

        return _to_img(fmri_data, func_type=func_type, mask_img=self.mask)  # type: ignore


def _extract_timing_pinel_motor(
    fps_onset: List[str], fps_duration: List[str], condition: str
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
        raise ValueError(
            f"No onset files found for condition: {condition} in pinel task"
        )
    elif len(fp_condition_onset) > 1:
        raise ValueError(
            f"Multiple onset files found for condition: {condition} in pinel task"
        )
    # load onset files for condition
    with open(fp_condition_onset[0], "r") as f:
        c_onsets = [float(o) for o in f.read().strip().split(" ")]

    # find duration files for condition
    fp_condition_duration = [fp for fp in fps_duration if condition in fp]
    if len(fp_condition_duration) == 0:
        raise ValueError(
            f"No duration files found for condition: {condition} in pinel task"
        )
    elif len(fp_condition_duration) > 1:
        raise ValueError(
            f"Multiple duration files found for condition: {condition} in pinel task"
        )
    # load duration files for condition
    with open(fp_condition_duration[0], "r") as f:
        c_durations = [float(d) for d in f.read().strip().split(" ")]

    # extract timing information from file paths for pinel task
    return c_onsets, c_durations


def _extract_timing_simon(
    fps_onset: List[str], condition: str
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
        raise ValueError(
            f"No onset files found for condition: {condition} in simon task"
        )
    elif len(fps_condition_onset) == 1:
        raise ValueError(
            f"Only one onset file found for condition: {condition} in simon task. Two files expected (correct and incorrect conditions)."
        )
    elif len(fps_condition_onset) > 2:
        raise ValueError(
            f"More than two onset files found for condition: {condition} in simon task"
        )

    # load onset files for correct and incorrect conditions
    c_onsets = []
    c_durations = []
    for fp in fps_condition_onset:
        with open(fp, "r") as f:
            c_onset_duration = [o.split(":") for o in f.read().strip().split(" ")]
            try:
                c_onsets.extend(
                    [float(od[0]) for od in c_onset_duration if float(od[0]) >= 0]
                )
                c_durations.extend(
                    [float(od[1]) for od in c_onset_duration if float(od[0]) >= 0]
                )
            except IndexError:
                raise ValueError(
                    f"Onset file for condition: {condition} in simon task does not have correct format."
                    " Each value should have onset and duration separated by a colon ':'."
                )

    # extract timing information from file paths for simon task
    return c_onsets, c_durations
