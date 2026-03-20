"""
Perform full analysis pipeline on selected subject
"""

import argparse
import os
import pickle

from typing import Literal

import nibabel as nib
from task_arousal.analysis.pca import PCA
from task_arousal.analysis.dlm import (
    DistributedLagPhysioModel,
    DistributedLagEventModel,
)

from task_arousal.dataset.dataset_euskalibur import (
    DatasetEuskalibur,
    PINEL_CONDITIONS,
    SIMON_CONDITIONS,
    MOTOR_CONDITIONS as MOTOR_CONDITIONS_EUSKALIBUR,
)
from task_arousal.dataset.dataset_nsd import DatasetNsd, NSDIMAGERY_CONDITIONS

from task_arousal.dataset.dataset_utils import DatasetLoad
from task_arousal.constants import (
    TR_EUSKALIBUR,
)

# define output directory
OUT_DIRECTORY = "results"

# physio signal labels
PHYSIO_LABELS_EUSKALIBUR = [
    "ppg_amplitude",
    "heart_rate",
    "resp_amp",
    "resp_rate",
    "endtidal_co2",
    "endtidal_o2",
]
PHYSIO_LABELS_NSD = [
    "ppg_amplitude",
    "heart_rate",
    "resp_amp",
    "resp_rate",
]

# define all tasks (exclude Motor task)
TASKS_EUSKALIBUR = ["pinel", "simon", "motor", "rest", "breathhold"]
# define tasks with event conditions
TASKS_EVENT_EUSKALIBUR = ["pinel", "simon", "motor"]

# define NSD tasks (just rest task right now)
TASKS_NSD = ["rest", "nsdimagery"]
# define NSD tasks weith event conditions
TASKS_EVENT_NSD = ["nsdimagery"]

# define analyses to perform
ANALYSES = ["dlm_physio", "dlm_event", "pca"]


# define Dataset type
Dataset = DatasetEuskalibur | DatasetNsd


def main(
    dataset: Literal["euskalibur", "nsd"],
    subject: str | None,
    analysis: str | None,
    task: str | None,
    space: Literal["surface", "volume"] = "volume",
    me_type: Literal["optcomb", "t2", "s0"] = "optcomb",
) -> None:
    """
    Perform full analysis pipeline on selected subject

    Parameters
    ----------
    dataset : Literal["euskalibur", "nsd"]
        Dataset to perform analysis pipeline on
    subject : str | None
        Subject to perform analysis pipeline on.
    analysis : str | None
        Type of analysis to perform
    task : str | None
        Task to perform analysis on
    space : Literal["surface", "volume"]
        Space to write output in (surface or volume)
    me_type : Literal["optcomb", "t2", "s0"]
        Type of multi-echo data to load (optcomb, t2, or s0). Only relevant for volume data in the EuskalIBUR dataset.
        Ignored for surface data and NSD dataset.
    """
    # check inputs
    if space == "surface" and dataset != "euskalibur":
        raise ValueError("Surface space is only available for the EuskalIBUR dataset.")
    # me_type is only relevant for volume data in the EuskalIBUR dataset, if surface, ignore me_type
    if space == "surface" and me_type != "optcomb":
        print(
            "optimally combined data is only available for surface, ignoring me_type and loading surface data."
        )
        me_type = "optcomb"
    # me_type data is not available for NSD dataset, if dataset is NSD, ignore me_type
    if dataset == "nsd" and me_type != "optcomb":
        print(
            "Multi-echo data is not available for NSD dataset, ignoring me_type and loading data."
        )
        me_type = "optcomb"
    # initialize dataset loader
    if dataset == "euskalibur":
        if subject is None:
            raise ValueError("Subject must be specified for EuskalIBUR dataset")
        ds = DatasetEuskalibur(subject=subject)

        if task is not None:
            if task not in TASKS_EUSKALIBUR:
                raise ValueError(f"Task {task} not recognized for EuskalIBUR dataset")
            tasks = [task]
            if task in TASKS_EVENT_EUSKALIBUR:
                tasks_event = [task]
            else:
                tasks_event = []
        else:
            tasks = TASKS_EUSKALIBUR
            tasks_event = TASKS_EVENT_EUSKALIBUR
        # create dict mapping task to TR - this is the same for each task in EuskalIBUR dataset
        tr = {task: TR_EUSKALIBUR for task in tasks}
        physio_labels = PHYSIO_LABELS_EUSKALIBUR
        # For EuskalIBUR, subject is guaranteed to be non-None
        _subject: str = subject
        # create output directory if it doesn't exist
        os.makedirs(OUT_DIRECTORY + "/euskalibur", exist_ok=True)
    elif dataset == "nsd":
        if subject is None:
            raise ValueError("Subject must be specified for NSD dataset")
        ds = DatasetNsd(subject=subject)
        if task is not None:
            if task not in TASKS_NSD:
                raise ValueError(f"Task {task} not recognized for NSD dataset")
            tasks = [task]
            tasks_event = [task] if task in TASKS_EVENT_NSD else []
        else:
            tasks = TASKS_NSD
            tasks_event = TASKS_EVENT_NSD
        # TR is different for each task in NSD dataset, so we will handle TR in the file mapper class rather than as a constant
        tr = {task: ds.file_mapper.get_tr(task) for task in tasks}

        physio_labels = PHYSIO_LABELS_NSD
        # For NSD, subject is guaranteed to be non-None
        _subject: str = subject
        # create output directory if it doesn't exist
        os.makedirs(OUT_DIRECTORY + "/nsd", exist_ok=True)
    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    # if analysis is specified, only perform that analysis
    if analysis is not None:
        print(f"Performing only {analysis} analysis for subject {_subject}")
        _analysis = [analysis]
    else:
        _analysis = ANALYSES

    # perform PCA or DLM with physiological regressors for all tasks,
    # including those without event conditions using concatenated data
    if any(a in _analysis for a in ["pca", "dlm_physio"]):
        for task in tasks:
            print(
                f"Loading concatenated data for dataset {dataset}, subject {_subject}, task {task}"
            )
            data: DatasetLoad = ds.load_data(
                task=task, func_type=space, concatenate=True, me_type=me_type
            )  # type: ignore

            # perform PCA analysis
            if "pca" in _analysis:
                _pca(dataset, data, ds, _subject, task, space, me_type)
                print(
                    f"PCA analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )
            # perform DLM analysis with physiological regressors for tasks with physio signals
            # the NSDimagery dataset does not have physiological signals, so we will skip DLM with physio analysis for NSD dataset
            if dataset == "nsd" and task == "nsdimagery":
                print(
                    f"Skipping DLM with physiological regressors analysis for dataset {dataset}, subject {_subject}, task {task} since NSD dataset does not have physiological signals"
                )
            elif "dlm_physio" in _analysis and physio_labels is not None:
                _dlm_physio(
                    dataset,
                    data,
                    ds,
                    tr[task],
                    physio_labels,
                    _subject,
                    task,
                    space,
                    me_type,
                )
                print(
                    f"DLM with physiological regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )

    # only perform DLM analyses for tasks with event conditions
    if any(a in _analysis for a in ["dlm_event"]):
        for task in tasks_event:
            print(
                f"Loading data for dataset {dataset}, subject {_subject}, task {task} for DLM with event analyses"
            )
            data: DatasetLoad = ds.load_data(
                task=task, func_type=space, concatenate=False, me_type=me_type
            )  # type: ignore

            if "dlm_event" in _analysis:
                # perform DLM analysis with event regressors
                _dlm_event(dataset, data, ds, tr[task], _subject, task, space, me_type)
                print(
                    f"DLM with event regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )


def _dlm_event(
    dataset: str,
    data: DatasetLoad,
    ds: Dataset,
    tr: float,
    subject: str,
    task: str,
    space: Literal["volume", "surface"],
    me_type: Literal["optcomb", "t2", "s0"] = "optcomb",
) -> None:
    """
    Perform Distributed Lag Model (DLM) analysis with event regressors
    on the given data and save results to files.

    Parameters
    ----------
    dataset : str
        Dataset type ('euskalibur')
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    tr: float
        Repetition time (TR) of fMRI data
    subject : str
        Subject identifier
    task : str
        Task identifier
    space : Literal["volume", "surface"]
        Space to write output in (surface or volume)
    me_type : Literal["optcomb", "t2", "s0"]
        Type of multi-echo data to load (optcomb, t2, or s0). Special suffix used if t2 or s0 data, otherwise ignored.
    """
    print(
        f"Performing DLM with event regressors on dataset {dataset}, subject {subject}, task {task}"
    )
    if dataset == "euskalibur":
        if task == "pinel":
            conditions = PINEL_CONDITIONS
        elif task == "simon":
            conditions = SIMON_CONDITIONS
        elif task == "motor":
            conditions = MOTOR_CONDITIONS_EUSKALIBUR
        else:
            raise ValueError(f"Task {task} not recognized for EuskalIBUR dataset")
    elif dataset == "nsd":
        if task == "nsdimagery":
            conditions = NSDIMAGERY_CONDITIONS
        else:
            raise ValueError(f"Task {task} not recognized for NSD dataset")
    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    if me_type in ["t2", "s0"]:
        suffix = "_" + me_type
    else:
        suffix = ""

    # estimate DLM with event regressors with default parameters
    dlm = DistributedLagEventModel(tr=tr)
    dlm = dlm.fit(
        event_dfs=data["events"],
        outcome_data=data["fmri"],  # type: ignore
    )
    # loop through conditions and write predicted functional time courses to nifti files
    for condition in conditions:
        dlm_eval = dlm.evaluate(trial=condition)
        pred_func_img = ds.to_img(dlm_eval.pred_outcome, func_type=space)
        nib.save(  # type: ignore
            pred_func_img,
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_event_{condition}{suffix}{'.nii.gz' if space == 'volume' else '.dtseries.nii'}",
        )
        # write dlm metadata (including betas, t-stats, etc.) to pickle file
        pickle.dump(
            dlm_eval,
            open(
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_event_{condition}_metadata{suffix}.pkl",
                "wb",
            ),
        )


def _dlm_physio(
    dataset: str,
    data: DatasetLoad,
    ds: Dataset,
    tr: float,
    physio_labels: list[str],
    subject: str,
    task: str,
    space: Literal["volume", "surface"],
    me_type: Literal["optcomb", "t2", "s0"] = "optcomb",
) -> None:
    """
    Perform Distributed Lag Model (DLM) analysis with physiological regressors
    on the given data and save results to files.

    Parameters
    ----------
    dataset : str
        Dataset type ('euskalibur')
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    tr: float
        Repetition time (TR) of fMRI data
    physio_labels: list[str]
        List of physiological signal labels
    subject : str
        Subject identifier
    task : str
        Task identifier
    space : Literal["volume", "surface"]
        Space to write output in (surface or volume)
    me_type : Literal["optcomb", "t2", "s0"]
        Type of multi-echo data to load (optcomb, t2, or s0).
        Special suffix used if t2 or s0 data, otherwise ignored.
    """
    print(
        f"Performing DLM with physiological regressors on subject {subject}, task {task}"
    )
    if me_type in ["t2", "s0"]:
        suffix = "_" + me_type
    else:
        suffix = ""
    # loop through physio signals
    for physio_label in physio_labels:
        # estimate DLM with physiological regressors
        # fix number of knots
        dlm = DistributedLagPhysioModel(
            tr=tr, neg_nlags=-15, nlags=15, n_knots=5, basis_type="cr"
        )
        dlm = dlm.fit(
            X=data["physio"][0][physio_label].to_numpy().reshape(-1, 1),
            Y=data["fmri"][0],  # type: ignore
        )
        # estimate functional time courses at each voxel to lagged physio signal
        dlm_eval = dlm.evaluate()
        # write predicted functional time courses to nifti file
        pred_func_img = ds.to_img(dlm_eval.pred_outcome, func_type=space)
        nib.save(  # type: ignore
            pred_func_img,
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_physio_{physio_label}{suffix}{'.nii.gz' if space == 'volume' else '.dtseries.nii'}",
        )

        # write dlm metadata to pickle file
        pickle.dump(
            dlm_eval,
            open(
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_physio_{physio_label}_metadata{suffix}.pkl",
                "wb",
            ),
        )


def _pca(
    dataset: str,
    data: DatasetLoad,
    ds: Dataset,
    subject: str | None,
    task: str,
    space: Literal["volume", "surface"],
    me_type: Literal["optcomb", "t2", "s0"] = "optcomb",
) -> None:
    """
    Perform PCA decomposition on fMRI data and save results to files

    Parameters
    ----------
    dataset : str
        Dataset identifier
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    subject : str
        Subject identifier
    task : str
        Task identifier
    space : Literal["volume", "surface"]
        Space to write output in (surface or volume)
    me_type : Literal["optcomb", "t2", "s0"]
        Type of multi-echo data to load (optcomb, t2, or s0). Special suffix
        used if t2 or s0 data, otherwise ignored.
    """
    print(f"Performing PCA on dataset {dataset}, subject {subject}, task {task}")
    # estimate PCA with 10 components
    pca = PCA(n_components=10)
    # run PCA decomposition
    pca_results = pca.decompose(data["fmri"][0])
    # write loadings to nifti file
    pca_loadings = ds.to_img(pca_results.loadings.T, func_type=space)
    if me_type in ["t2", "s0"]:
        suffix = "_" + me_type
    else:
        suffix = ""
    nib.save(  # type: ignore
        pca_loadings,
        f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_pca_loadings{suffix}{'.nii.gz' if space == 'volume' else '.dtseries.nii'}",
    )
    # write pca metadata (including pc scores, exp var, etc.) to pickle file
    pickle.dump(
        pca_results,
        open(
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_pca_metadata{suffix}.pkl",
            "wb",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform full analysis pipeline on selected subject"
    )
    # add dataset argument
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["euskalibur", "nsd"],
        help="Dataset to perform analysis pipeline on",
    )
    # add subject argument
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=True,
        help="Subject to perform preprocessing pipeline. "
        "For BIDS datasets (euskalibur), only the subject ID is needed, e.g., 001 (not sub-001). "
        "For NSD, the full subject ID is needed (e.g. subj01).",
    )
    # add optional analysis argument (default: all analyses)
    parser.add_argument(
        "-a",
        "--analysis",
        type=str,
        choices=ANALYSES,
        required=False,
        default=None,
        help="Type of analysis to perform",
    )
    # add optional task argument (default: all tasks)
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=False,
        default=None,
        help="Task to perform analysis on",
    )
    # add optional argument to specify what space (surface or volume) to write output in
    parser.add_argument(
        "-p",
        "--space",
        type=str,
        choices=["surface", "volume"],
        required=False,
        default="volume",
        help="Space to write output in (surface or volume). Surface space is only available for the EuskalIBUR dataset.",
    )
    # add optional argument so specify the multi-echo data type to load (optcomb, t2, or s0), only relevant for volume data in the EuskalIBUR dataset
    parser.add_argument(
        "-m",
        "--me_type",
        type=str,
        choices=["optcomb", "t2", "s0"],
        required=False,
        default="optcomb",
        help="Type of multi-echo data to load (optcomb, t2, or s0). Only relevant for volume data in the EuskalIBUR dataset. Ignored for surface data and NSD dataset.",
    )
    # parse arguments
    args = parser.parse_args()
    main(args.dataset, args.subject, args.analysis, args.task, args.space, args.me_type)
