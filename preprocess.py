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
    me_type: list[Literal["optcomb", "t2", "s0"]] = ["optcomb"],
    echo_pipeline: bool = False,
    skip_physio: bool = False,
    skip_func: bool = False,
):
    """Perform full preprocessing pipeline on selected subject or all subjects."""
    # only allow echo_pipeline for euskalibur dataset since NSD does not have multi-echo data
    if echo_pipeline and dataset != "euskalibur":
        raise ValueError(
            "Echo pipeline can only be used for the Euskalibur dataset since NSD does not have multi-echo data."
        )
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
                    echo_pipeline=echo_pipeline,
                    func_type=func_type,
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
        nargs="+",
        choices=["optcomb", "t2", "s0"],
        default=["optcomb"],
        help="One or more multi-echo data types to preprocess. Allowed values are "
        "'optcomb', 't2', and 's0'. Defaults to 'optcomb'.",
    )
    parser.add_argument(
        "-echo_pipeline",
        "--echo_pipeline",
        action="store_true",
        required=False,
        default=False,
        help="For the Euskalibut dataset. Whether to estimate T2* and S0 from multi-echo fMRI data using a log-linear fit "
        "and use the estimated T2* and S0 values for preprocessing instead of the raw echo data. Defaults to False.",
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

    args = parser.parse_args()
    main(
        args.dataset,
        args.subject,
        args.task,
        args.func_type,
        args.me_type,
        args.echo_pipeline,
        args.skip_physio,
        args.skip_func,
    )
