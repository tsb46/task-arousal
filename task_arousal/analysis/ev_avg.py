"""
Event-averaged analysis function.
"""

import warnings

from dataclasses import dataclass


import pandas as pd
import numpy as np


@dataclass
class EventAverageResults:
    avg_responses: dict[str, np.ndarray]
    events: dict[
        str, list[tuple[float, float]]
    ]  # trial_type -> list of (onset, duration)
    trial_types: list[str]


def event_average(
    tr: float,
    event_dfs: list[pd.DataFrame],
    fmri_data: list[np.ndarray],
    duration_extend: int = 5,
) -> EventAverageResults:
    """
    Compute event-averaged fMRI response for each trial type.

    Parameters
    ----------
    tr : float
        Repetition time of the fMRI data.
    event_dfs : list of pd.DataFrame
        List of event dataframes for each run.
    fmri_data : list of np.ndarray
        List of fMRI data arrays for each run.
    duration_extend : int, optional
        Additional duration to extend the event window in TRs, by default 5.
        This can help capture some of the post-event responses.

    Returns
    -------
    EventAverageResults
        Event-averaged fMRI response for each trial type.
    """

    # get trial types from all event dfs
    trial_types = []
    for i, event_df in enumerate(event_dfs):
        unique_trials = event_df["trial_type"].unique().tolist()
        for trial in unique_trials:
            if trial not in trial_types:
                if i > 0:
                    warnings.warn(
                        f"Adding new trial type '{trial}' from dataframe {i} that was not in the first dataframe."
                    )
                trial_types.append(trial)

    # initialize dicts to hold all events and responses
    all_events = {trial: [] for trial in trial_types}
    all_responses = {trial: [] for trial in trial_types}
    # loop over event dfs and fmri data
    for event_df, fmri in zip(event_dfs, fmri_data):
        for trial_type in event_df["trial_type"].unique():
            trial_events = event_df[event_df["trial_type"] == trial_type]
            onsets = trial_events["onset"].values
            durations = trial_events["duration"].values

            for onset, duration in zip(onsets, durations):
                # onset and duration are in seconds, convert to indices
                onset_idx = int(onset / tr)
                duration_idx = int(duration / tr) + duration_extend
                response_segment = fmri[onset_idx : onset_idx + duration_idx]
                all_events[trial_type].append((onset, duration))
                all_responses[trial_type].append(response_segment)

    # compute average response across all trial types
    avg_responses = {}
    for trial_type in trial_types:
        avg_responses[trial_type] = np.mean(np.array(all_responses[trial_type]), axis=0)

    return EventAverageResults(
        avg_responses=avg_responses,
        events=all_events,
        trial_types=trial_types,
    )
