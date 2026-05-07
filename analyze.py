"""
Perform full analysis pipeline on selected subject
"""

import argparse
import os
import pickle

from typing import Literal

from bidsschematools import data
import nibabel as nib
import numpy as np
from scipy.stats import zscore

from task_arousal.analysis.pca import PCA, GroupPCA
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
from task_arousal.constants import TR_EUSKALIBUR, ECHOS_EUSKALIBUR

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
ANALYSES = ["dlm_physio", "dlm_event", "pca", "group_pca"]


# define Dataset type
Dataset = DatasetEuskalibur | DatasetNsd


def main(
    dataset: Literal["euskalibur", "nsd"],
    subject: str | None,
    analysis: str | None,
    task: str | None,
    space: Literal["surface", "volume"] = "volume",
    me_type: Literal["optcomb", "echo"] = "optcomb",
    physio_regressor: str | None = None,
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
    me_type : Literal["optcomb", "echo"] = "optcomb"
        Type of multi-echo data to load (optcomb or echo). Only relevant for volume data in the EuskalIBUR dataset. If
        "echo" is selected, all echo data will be loaded and analysis will be performed on each echo. If "optcomb" is selected, optimally combined data will be loaded and standard DLM analyses will be performed.
        Ignored for surface data and NSD dataset.
    physio_regressor : str | None
        Restrict DLM with physiological regressors analysis to specific physiological regressor. Only relevant if performing dlm_physio analysis.
        If not specified, DLM with physiological regressors analysis will be performed for all available physiological regressors.
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
        if physio_regressor is not None:
            if physio_regressor not in PHYSIO_LABELS_EUSKALIBUR:
                raise ValueError(
                    f"Physiological regressor {physio_regressor} not recognized for EuskalIBUR dataset"
                )
            physio_labels = [physio_regressor]
        else:
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

        if physio_regressor is not None:
            if physio_regressor not in PHYSIO_LABELS_NSD:
                raise ValueError(
                    f"Physiological regressor {physio_regressor} not recognized for NSD dataset"
                )
            physio_labels = [physio_regressor]
        else:
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

    # perform PCA for all tasks
    if any(a in _analysis for a in ["pca"]) and me_type != "echo":
        for task in tasks:
            print(
                f"Loading concatenated data for dataset {dataset}, subject {_subject}, task {task}"
            )
            data: DatasetLoad = ds.load_data(
                task=task, func_type=space, concatenate=True, me_type="optcomb"
            )  # type: ignore

            # perform PCA analysis
            if "pca" in _analysis:
                _pca(dataset, data, ds, _subject, task, space, me_type="optcomb")
                print(
                    f"PCA analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )

    # perform DLM analyses for tasks with event conditions and with physiological signals
    if (
        any(a in _analysis for a in ["dlm_event", "dlm_physio"])
        and me_type == "optcomb"
    ):
        for task in tasks_event:
            print(
                f"Loading data for dataset {dataset}, subject {_subject}, task {task} for DLM with event analyses"
            )
            data: DatasetLoad = ds.load_data(
                task=task, func_type=space, concatenate=False, me_type="optcomb"
            )  # type: ignore

            if "dlm_event" in _analysis:
                # perform DLM analysis with event regressors
                _dlm_event(
                    dataset,
                    data,
                    ds,
                    tr[task],
                    _subject,
                    task,
                    space,
                    me_type="optcomb",
                )
                print(
                    f"DLM with event regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )

        for task in tasks:
            # perform DLM analysis with physiological regressors for tasks with physio signals
            # the NSDimagery dataset does not have physiological signals, so we will skip DLM with physio analysis for NSD dataset
            if dataset == "nsd" and task == "nsdimagery":
                print(
                    f"Skipping DLM with physiological regressors analysis for dataset {dataset}, subject {_subject}, task {task} since NSD dataset does not have physiological signals"
                )
            elif "dlm_physio" in _analysis:
                data: DatasetLoad = ds.load_data(
                    task=task, func_type=space, concatenate=False, me_type="optcomb"
                )  # type: ignore
                _dlm_physio(
                    dataset,
                    data,
                    ds,
                    tr[task],
                    physio_labels,
                    _subject,
                    task,
                    space,
                    me_type="optcomb",
                )
                print(
                    f"DLM with physiological regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )

    # perform group PCA across echoes (multi-echo data only, euskalibur only)
    if "group_pca" in _analysis and me_type == "echo" and dataset == "euskalibur":
        if not isinstance(ds, DatasetEuskalibur):
            raise ValueError(
                "Multi-echo analyses are only supported for the EuskalIBUR dataset"
            )
        for task in tasks:
            print(
                f"Performing Group PCA for dataset {dataset}, subject {_subject}, task {task}"
            )
            _group_pca(dataset, ds, _subject, task)
            print(
                f"Group PCA complete for dataset {dataset}, subject {_subject}, task {task}"
            )

    # perform individual echo DLM analyses for multi-echo data
    if any(a in _analysis for a in ["dlm_event", "dlm_physio"]) and me_type == "echo":
        for task in tasks_event:
            print(
                f"Streaming multi-echo data for dataset {dataset}, subject {_subject}, task {task} for DLM analyses"
            )
            if "dlm_event" in _analysis:
                _dlm_event_echo(dataset, ds, tr[task], _subject, task)
                print(
                    f"DLM with event regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )
        for task in tasks:
            if "dlm_physio" in _analysis:
                _dlm_physio_echo(dataset, ds, tr[task], physio_labels, _subject, task)
                print(
                    f"DLM with physiological regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}"
                )


def _group_pca(
    dataset: str,
    ds: DatasetEuskalibur,
    subject: str,
    task: str,
    n_individual_components: int = 50,
    n_group_components: int = 10,
) -> None:
    """
    Perform Group PCA across echoes for a given task.

    Each echo is loaded one at a time and its runs are temporally concatenated
    before fitting an individual PCA. This avoids holding the full multi-echo
    dataset in memory simultaneously. The per-echo PCAResults are then passed
    to GroupPCA to estimate shared latent components.

    Per-echo spatial projection maps (encoders, shape (n_components, n_voxels))
    are written as NIfTI files. The full GroupPCAResults object is saved as a
    pickle.

    Parameters
    ----------
    dataset : str
        Dataset identifier, used for output path.
    ds : DatasetEuskalibur
        Dataset object for loading data and writing images.
    subject : str
        Subject identifier.
    task : str
        Task identifier.
    n_individual_components : int
        Number of components for each per-echo individual PCA. Default 50.
    n_group_components : int
        Number of shared components for the group PCA. Default 10.
    """
    pca = PCA(n_components=n_individual_components)
    pca_results = []

    # fit individual PCA per echo, loading one echo at a time
    for echo_index in range(len(ECHOS_EUSKALIBUR)):
        data = ds.load_data(
            task=task,
            me_type="echo",
            concatenate=False,
            normalize=False,
            load_physio=False,
            echo_n=echo_index,
        )
        # temporally concatenate runs before PCA
        X = np.concatenate(data["fmri"], axis=0)  # (n_timepoints_total, n_voxels)
        result = pca.decompose(X)
        pca_results.append(result)
        print(
            f"  Individual PCA complete for echo {echo_index + 1}/{len(ECHOS_EUSKALIBUR)}"
        )

        # --- DEBUG: save per-echo individual PCA results ---
        # save PCAResults as pickle
        pickle.dump(
            result,
            open(
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_individual_pca_echo_{echo_index + 1}_metadata.pkl",
                "wb",
            ),
        )
        # save loadings (n_features, n_components) as NIfTI brain map
        loadings_img = ds.to_img(result.loadings.T)
        nib.save(  # type: ignore
            loadings_img,
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_individual_pca_echo_{echo_index + 1}_loadings.nii.gz",
        )

    # --- DEBUG: compute and save per-echo diagnostics before group PCA ---
    echo_diagnostics = []
    for echo_index, result in enumerate(pca_results):
        # U orthonormality check
        gram = result.U.T @ result.U
        max_off_diag = float(np.abs(gram - np.eye(gram.shape[0])).max())
        # Va row norms (voxel contribution to top PCs)
        va_row_norms = (result.Va**2).sum(axis=0)
        diag = {
            "echo_index": echo_index,
            # singular value spectrum
            "s": result.s,
            "s_0": float(result.s[0]),
            "s_last": float(result.s[-1]),
            # pc_scores vs U variance — pc_scores encodes singular values, U does not
            "pc_scores_std": float(result.pc_scores.std()),
            "U_std": float(result.U.std()),
            # voxel-space Va norms — heavy tail indicates edge/susceptibility voxels dominating
            "va_row_norms_mean": float(va_row_norms.mean()),
            "va_row_norms_99th": float(np.percentile(va_row_norms, 99)),
            "va_row_norms_max": float(va_row_norms.max()),
            # U orthonormality
            "U_gram_max_off_diag": max_off_diag,
        }
        echo_diagnostics.append(diag)
        print(
            f"  Echo {echo_index + 1} diagnostics: s[0]={diag['s_0']:.2f}, "
            f"pc_scores_std={diag['pc_scores_std']:.4f}, U_std={diag['U_std']:.4f}, "
            f"Va_norm_99th={diag['va_row_norms_99th']:.4f}, U_gram_off_diag={max_off_diag:.2e}"
        )

    pickle.dump(
        echo_diagnostics,
        open(
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_group_pca_echo_diagnostics.pkl",
            "wb",
        ),
    )

    # fit group PCA on concatenated PC scores across echoes
    gpca = GroupPCA(n_components=n_group_components)
    group_result = gpca.decompose(pca_results)

    # --- DEBUG: projection/embedding norms across echoes ---
    proj_emb_diagnostics = []
    for echo_index, (proj, emb) in enumerate(
        zip(group_result.individual_projections, group_result.individual_embeddings)
    ):
        diag = {
            "echo_index": echo_index,
            "projection_row_norm_mean": float(np.linalg.norm(proj, axis=1).mean()),
            "projection_row_norm_std": float(np.linalg.norm(proj, axis=1).std()),
            "embedding_col_norm_mean": float(np.linalg.norm(emb, axis=0).mean()),
            "embedding_col_norm_std": float(np.linalg.norm(emb, axis=0).std()),
        }
        proj_emb_diagnostics.append(diag)
        print(
            f"  Echo {echo_index + 1} projection norm={diag['projection_row_norm_mean']:.4f}, "
            f"embedding norm={diag['embedding_col_norm_mean']:.4f}"
        )

    pickle.dump(
        proj_emb_diagnostics,
        open(
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_group_pca_proj_emb_diagnostics.pkl",
            "wb",
        ),
    )

    # save per-echo spatial projection maps as NIfTI
    # individual_projections[i]: (n_components, n_voxels) — to_img expects (n_timepoints, n_voxels)
    for echo_index, projection in enumerate(group_result.individual_projections):
        proj_img = ds.to_img(projection)
        nib.save(  # type: ignore
            proj_img,
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_group_pca_projection_echo_{echo_index + 1}.nii.gz",
        )

    # save full GroupPCAResults as pickle
    pickle.dump(
        group_result,
        open(
            f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_group_pca_metadata.pkl",
            "wb",
        ),
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


def _dlm_event_echo(
    dataset: str,
    ds: Dataset,
    tr: float,
    subject: str,
    task: str,
) -> None:
    """
    Perform Distributed Lag Model (DLM) analysis with event regressors
    on each echo individually.

    Parameters
    ----------
    dataset : str
        Dataset type ('euskalibur')
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
    if task == "pinel":
        conditions = PINEL_CONDITIONS
    elif task == "simon":
        conditions = SIMON_CONDITIONS
    elif task == "motor":
        conditions = MOTOR_CONDITIONS_EUSKALIBUR
    else:
        raise ValueError(f"Task {task} not recognized for EuskalIBUR dataset")

    if not isinstance(ds, DatasetEuskalibur):
        raise ValueError(
            "Multi-echo analyses are only supported for the EuskalIBUR dataset"
        )

    # loop through individual echoes and perform DLM analysis for each echo separately
    for echo_index in range(len(ECHOS_EUSKALIBUR)):
        # load echo data given index
        data = ds.load_data(
            task=task,
            me_type="echo",
            concatenate=False,
            normalize=True,
            load_physio=False,
            echo_n=echo_index,
        )
        # estimate DLM with event regressors with default parameters
        dlm = DistributedLagEventModel(tr=tr)
        dlm = dlm.fit(
            event_dfs=data["events"],
            outcome_data=data["fmri"],
        )
        # loop through conditions and write predicted functional time courses to nifti files
        for condition in conditions:
            dlm_eval = dlm.evaluate(trial=condition)
            pred_func_img = ds.to_img(dlm_eval.pred_outcome)
            nib.save(  # type: ignore
                pred_func_img,
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_event_{condition}_echo_{echo_index + 1}.nii.gz",
            )
            # write dlm metadata (including betas, t-stats, etc.) to pickle file
            pickle.dump(
                dlm_eval,
                open(
                    f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_event_{condition}_echo_{echo_index + 1}_metadata.pkl",
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
            X=[run[physio_label].to_numpy().reshape(-1, 1) for run in data["physio"]],  # type: ignore
            Y=data["fmri"],  # type: ignore[arg-type]
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


def _dlm_physio_echo(
    dataset: str,
    ds: Dataset,
    tr: float,
    physio_labels: list[str],
    subject: str,
    task: str,
) -> None:
    """
    Perform Distributed Lag Model (DLM) analysis with physiological regressors
    on each echo individually.

    Parameters
    ----------
    dataset : str
        Dataset type ('euskalibur')
    ds : Dataset
        Dataset object for handling data operations
    tr: float
        Repetition time (TR) of fMRI data
    physio_labels: list[str]
        List of physiological signal labels to analyze
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(
        f"Performing DLM with physiological regressors on dataset {dataset}, subject {subject}, task {task}"
    )
    if not isinstance(ds, DatasetEuskalibur):
        raise ValueError(
            "Multi-echo analyses are only supported for the EuskalIBUR dataset"
        )

    # loop through individual echoes and perform DLM analysis for each echo separately
    for echo_index in range(len(ECHOS_EUSKALIBUR)):
        # load echo data given index
        data = ds.load_data(
            task=task,
            me_type="echo",
            concatenate=False,
            normalize=True,
            load_physio=True,
            echo_n=echo_index,
        )
        # perform DLM analysis with physiological regressors for each physio signal separately
        for physio_label in physio_labels:
            dlm = DistributedLagPhysioModel(
                tr=tr, neg_nlags=-15, nlags=15, n_knots=5, basis_type="cr"
            )
            dlm = dlm.fit(
                X=[
                    run[physio_label].to_numpy().reshape(-1, 1)
                    for run in data["physio"]
                ],  # type: ignore
                Y=data["fmri"],  # type: ignore[arg-type]
            )
            dlm_eval = dlm.evaluate()
            pred_func_img = ds.to_img(dlm_eval.pred_outcome)
            nib.save(  # type: ignore
                pred_func_img,
                f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_physio_{physio_label}_echo_{echo_index + 1}.nii.gz",
            )
            # . write dlm metadata to pickle file
            pickle.dump(
                dlm_eval,
                open(
                    f"{OUT_DIRECTORY}/{dataset}/sub-{subject}_{task}_dlm_physio_{physio_label}_echo{echo_index + 1}_metadata.pkl",
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
    # add optional argument so specify the multi-echo data type to load (optcomb, mono_exp, or echo), only relevant for volume data in the EuskalIBUR dataset
    parser.add_argument(
        "-m",
        "--me_type",
        type=str,
        choices=["optcomb", "echo"],
        required=False,
        default="optcomb",
        help="Type of multi-echo data to load (optcomb, echo). Only relevant for volume data in the EuskalIBUR dataset. Ignored for surface data and NSD dataset."
        "If 'echo' is selected, all echo data will be loaded and analyse performed on each echo. If 'optcomb' is selected, optimally combined data will be loaded and standard DLM analyses will be performed.",
    )
    # add optional argument to restrict dlm_physio analysis to specific physio regressor
    parser.add_argument(
        "-r",
        "--physio_regressor",
        type=str,
        choices=PHYSIO_LABELS_EUSKALIBUR,  # use physio labels from EuskalIBUR since they are more extensive, will just be ignored for NSD dataset
        required=False,
        default=None,
        help="Restrict DLM with physiological regressors analysis to specific physiological regressor. Only relevant if performing dlm_physio analysis. If not specified, DLM with physiological regressors analysis will be performed for all available physiological regressors.",
    )
    # parse arguments
    args = parser.parse_args()
    main(
        args.dataset,
        args.subject,
        args.analysis,
        args.task,
        args.space,
        args.me_type,
        args.physio_regressor,
    )
