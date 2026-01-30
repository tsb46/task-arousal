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
from task_arousal.dataset.dataset_pan import DatasetPan, PAN_CONDITIONS

from task_arousal.dataset.dataset_utils import DatasetLoad
from task_arousal.constants import (
    MASK_GM_EUSKALIBUR,
    TR_EUSKALIBUR,
    MASK_EUSKALIBUR,
    MASK_GM_PAN,
    TR_PAN,
    MASK_PAN,
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

# define all tasks (exclude Motor task)
TASKS_EUSKALIBUR = ["pinel", "simon", "motor", "rest", "breathhold"]
TASKS_PAN = [
    "audvisattn",
    "audviswm",
    "epiproj",
    "langlocaud",
    "langlocvis",
    "msit",
    "rest",
    "spatialwm",
    "tomfalse",
    "tompain",
    "verbalwm",
    "vmsit",
]
# define tasks with event conditions
TASKS_EVENT_EUSKALIBUR = ["pinel", "simon", "motor"]
TASKS_EVENT_PAN = [
    "audvisattn",
    "audviswm",
    "epiproj",
    "langlocaud",
    "langlocvis",
    "msit",
    "spatialwm",
    "tomfalse",
    "tompain",
    "verbalwm",
    "vmsit",
]

# define analyses to perform
ANALYSES = [
    "dlm_physio",
    "dlm_event",
    "pca",
]

# define Dataset type
Dataset = DatasetEuskalibur | DatasetPan


def main(
    dataset: Literal["euskalibur", "pan"],
    subject: str | None,
    analysis: str | None,
    task: str | None,
) -> None:
    """
    Perform full analysis pipeline on selected subject

    Parameters
    ----------
    dataset : Literal["euskalibur", "pan"]
        Dataset to perform analysis pipeline on
    subject : str | None
        Subject to perform analysis pipeline on. Required if dataset is euskalibur or pan.
    analysis : str | None
        Type of analysis to perform
    task : str | None
        Task to perform analysis on
    """
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
        tr = TR_EUSKALIBUR
        mask = MASK_EUSKALIBUR
        mask_gm = MASK_GM_EUSKALIBUR
        physio_labels = PHYSIO_LABELS_EUSKALIBUR
        # For EuskalIBUR, subject is guaranteed to be non-None
        _subject: str = subject
        # create output directory if it doesn't exist
        os.makedirs(OUT_DIRECTORY + "/euskalibur", exist_ok=True)
    elif dataset == "pan":
        if subject is None:
            raise ValueError("Subject must be specified for PAN dataset")
        ds = DatasetPan(subject=subject)
        if task is not None:
            if task not in TASKS_PAN:
                raise ValueError(f"Task {task} not recognized for PAN dataset")
            tasks = [task]
            if task in TASKS_EVENT_PAN:
                tasks_event = [task]
            else:
                tasks_event = []
        else:
            tasks = TASKS_PAN
            tasks_event = TASKS_EVENT_PAN
        tr = TR_PAN
        mask = MASK_PAN
        mask_gm = MASK_GM_PAN
        physio_labels = None  # PAN dataset does not have physio signals
        # For PAN, subject is guaranteed to be non-None
        _subject: str = subject
        # create output directory if it doesn't exist
        os.makedirs(OUT_DIRECTORY + "/pan", exist_ok=True)
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
            data: DatasetLoad = ds.load_data(task=task, concatenate=True)  # type: ignore

            # perform PCA analysis
            if "pca" in _analysis:
                _pca(dataset, data, ds, _subject, task)
                print(
                    f"PCA analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )
            # perform DLM analysis with physiological regressors for tasks with physio signals
            if "dlm_physio" in _analysis and physio_labels is not None:
                _dlm_physio(dataset, data, ds, tr, physio_labels, _subject, task)
                print(
                    f"DLM with physiological regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )

    # only perform DLM and GLM physio analyses for tasks with event conditions
    if any(a in _analysis for a in ["dlm_event", "glm_physio", "dlm_ca", "pls", "rrr"]):
        for task in tasks_event:
            print(
                f"Loading data for dataset {dataset}, subject {_subject}, task {task} for DLM with event and GLM with physio analyses"
            )
            data: DatasetLoad = ds.load_data(task=task, concatenate=False)  # type: ignore

            if "dlm_event" in _analysis:
                # perform DLM analysis with event regressors
                _dlm_event(dataset, data, ds, tr, _subject, task)
                print(
                    f"DLM with event regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )


def _dlm_event(
    dataset: str, data: DatasetLoad, ds: Dataset, tr: float, subject: str, task: str
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
    else:
        if task in PAN_CONDITIONS:
            conditions = PAN_CONDITIONS[task]["conditions"]
        else:
            raise ValueError(f"Task {task} not recognized for PAN dataset")

    # estimate DLM with event regressors with default parameters
    dlm = DistributedLagEventModel(tr=tr)
    dlm = dlm.fit(
        event_dfs=data["events"],
        outcome_data=data["fmri"],  # type: ignore
    )
    # loop through conditions and write predicted functional time courses to nifti files
    for condition in conditions:
        dlm_eval = dlm.evaluate(trial=condition)
        pred_func_img = ds.to_4d(dlm_eval.pred_outcome)
        nib.nifti1.save(
            pred_func_img,
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_event_{condition}.nii.gz",
        )
        # write dlm metadata (including betas, t-stats, etc.) to pickle file
        pickle.dump(
            dlm_eval,
            open(
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_event_{condition}_metadata.pkl",
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
    """
    print(
        f"Performing DLM with physiological regressors on subject {subject}, task {task}"
    )
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
        pred_func_img = ds.to_4d(dlm_eval.pred_outcome)
        nib.nifti1.save(
            pred_func_img,
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_physio_{physio_label}.nii.gz",
        )

        # write dlm metadata to pickle file
        pickle.dump(
            dlm_eval,
            open(
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_physio_{physio_label}_metadata.pkl",
                "wb",
            ),
        )


def _pca(
    dataset: str, data: DatasetLoad, ds: Dataset, subject: str | None, task: str
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
    """
    print(f"Performing PCA on dataset {dataset}, subject {subject}, task {task}")
    # estimate PCA with 10 components
    pca = PCA(n_components=10)
    # run PCA decomposition
    pca_results = pca.decompose(data["fmri"][0])
    # write loadings to nifti file
    pca_loadings = ds.to_4d(pca_results.loadings.T)
    nib.nifti1.save(
        pca_loadings,
        f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_pca_loadings.nii.gz",
    )
    # write pca metadata (including pc scores, exp var, etc.) to pickle file
    pickle.dump(
        pca_results,
        open(f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_pca_metadata.pkl", "wb"),
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
        required=False,
        default="euskalibur",
        choices=["euskalibur"],
        help="Dataset to perform analysis pipeline on",
    )
    # add subject argument
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        default=None,
        required=False,
        help="Subject to perform analysis pipeline. Required if dataset is euskalibur.",
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
    # parse arguments
    args = parser.parse_args()
    main(args.dataset, args.subject, args.analysis, args.task)
