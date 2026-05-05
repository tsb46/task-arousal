"""
Perform full preprocessing pipeline on selected subject
"""

import argparse

from typing import Literal

from task_arousal.io.file import get_dataset_subjects
from task_arousal.preprocess.pipeline import PreprocessingPipeline


def main(
    dataset: Literal["euskalibur", "nsd"],
    subject: str | None = None,
    task: str | None = None,
    func_type: Literal["volume", "surface"] = "volume",
    me_type: Literal["optcomb", "t2s0", "echo"] = "optcomb",
    skip_me_fit: bool = True,
    skip_physio: bool = False,
    skip_func: bool = False,
    n_jobs: int = 1,
):
    """Perform full preprocessing pipeline on selected subject or all subjects."""
    # loop through tasks and preprocess
    if subject is None:
        subjects = get_dataset_subjects(dataset)
    else:
        subjects = [subject]

    # preprocess by subject for EuskaliBUR and NSD
    if dataset in ["euskalibur", "nsd"]:
        for subject in subjects:
            print(f"Starting preprocessing for subject: {subject}")
            pipeline = PreprocessingPipeline(dataset, subject)
            tasks_to_process = [task] if task is not None else pipeline.tasks
            for task in tasks_to_process:
                print(f"Preprocessing task: {task} for subject: {subject}")
                pipeline.preprocess(
                    task=task,
                    save_physio_figs=True,
                    skip_physio=skip_physio,
                    skip_func=skip_func,
                    me_type=me_type,
                    skip_me_fit=skip_me_fit,
                    func_type=func_type,
                    n_jobs=n_jobs,
                )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform preprocessing pipeline on selected subject"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["euskalibur", "nsd"],
        help="Dataset to perform preprocessing pipeline.",
    )
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=False,
        default=None,
        help="Subject to perform preprocessing pipeline. "
        "For BIDS datasets (euskalibur), only the subject ID is needed, e.g., 001 (not sub-001). "
        "For NSD, the full subject ID is needed (e.g. subj01). If not provided, "
        "the pipeline will be run for all subjects in the dataset.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=False,
        default=None,
        help="Task to perform preprocessing pipeline. If not provided, "
        "the pipeline will be run for all tasks for the subject.",
    )
    parser.add_argument(
        "-f",
        "--func_type",
        type=str,
        required=False,
        default="volume",
        help="Functional data type to preprocess - volume or surface. If not provided, "
        "the default is volume.",
    )
    parser.add_argument(
        "--me_type",
        choices=["optcomb", "t2s0", "echo"],
        type=str,
        required=False,
        default="optcomb",
        help="A multi-echo data type to preprocess. Allowed values are "
        "'optcomb', 't2s0', and 'echo'. 't2s0' preprocesses time-varying T2* and S0 estimates from a linear log fit. "
        "'Echo' preprocessing each individual echo separately. "
        "Defaults to 'optcomb'. Note, this argument is only relevant if the dataset has multi-echo "
        "data (e.g. Euskalibur).",
    )
    parser.add_argument(
        "-skip_me_fit",
        "--skip_me_fit",
        action="store_true",
        required=False,
        default=True,
        help="For the Euskalibut dataset. Whether to skip estimating T2* and S0 from multi-echo fMRI data using a log-linear fit. Use"
        "this option if you already estimated T2* and S0 and just want to re-run the preprocessing pipeline on the T2* and S0 estimates. If not provided, "
        "the default is True.",
    )

    parser.add_argument(
        "-skip_physio",
        "--skip_physio",
        action="store_true",
        required=False,
        default=False,
        help="Whether to skip physiological data preprocessing. If not provided, "
        "the default is False.",
    )
    parser.add_argument(
        "-skip_func",
        "--skip_func",
        action="store_true",
        required=False,
        default=False,
        help="Whether to skip functional data preprocessing. If not provided, "
        "the default is False.",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of parallel jobs for fMRI file processing. "
        "1 means sequential (default). -1 uses all available CPU cores.",
    )

    args = parser.parse_args()
    main(
        args.dataset,
        args.subject,
        args.task,
        args.func_type,
        args.me_type,
        args.skip_me_fit,
        args.skip_physio,
        args.skip_func,
        args.n_jobs,
    )
