"""
Perform full preprocessing pipeline on selected subject
"""

import argparse

from typing import Literal

from task_arousal.io.file import get_dataset_subjects
from task_arousal.preprocess.pipeline import PreprocessingPipeline


def main(
    dataset: Literal["euskalibur", "pan"],
    subject: str | None = None,
    task: str | None = None,
    func_type: Literal["volume", "surface"] = "volume",
    skip_physio: bool = False,
    skip_func: bool = False,
):
    """Perform full preprocessing pipeline on selected subject or all subjects."""
    # loop through tasks and preprocess
    if subject is None:
        subjects = get_dataset_subjects(dataset)
    else:
        subjects = [subject]

    # preprocess by subject for EuskaliBUR and PAN
    if dataset in ["euskalibur", "pan"]:
        for subject in subjects:
            print(f"Starting preprocessing for subject: {subject}")
            pipeline = PreprocessingPipeline(dataset, subject, func_type=func_type)
            tasks_to_process = [task] if task is not None else pipeline.tasks
            for task in tasks_to_process:
                print(f"Preprocessing task: {task} for subject: {subject}")
                pipeline.preprocess(
                    task=task,
                    save_physio_figs=True,
                    skip_physio=skip_physio,
                    skip_func=skip_func,
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
        choices=["euskalibur", "pan"],
        help="Dataset to perform preprocessing pipeline.",
    )
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=False,
        default=None,
        help="Subject to perform preprocessing pipeline. "
        "Only the subject ID is needed, e.g., 001. If not provided, "
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
        args.skip_physio,
        args.skip_func,
    )
