"""
Perform full preprocessing pipeline on selected subject
"""

import argparse

from typing import Literal

from task_arousal.io.file import get_dataset_subjects
from task_arousal.preprocess.pipeline import PreprocessingPipeline

# define tasks to preprocess (exclude Motor task)
EUSKALIBUR_TASKS = ['pinel', 'simon', 'rest', 'breathhold']
HCP_TASKS = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR',
             'RELATIONAL', 'SOCIAL', 'WM']

def main(dataset: Literal['hcp', 'euskalibur'], subject: str | None = None):
    """Perform full preprocessing pipeline on selected subject or all subjects."""
    # loop through tasks and preprocess
    if subject is None:
        subjects = get_dataset_subjects(dataset)
    else:
        subjects = [subject]
    
    # select tasks based on dataset
    if dataset == 'hcp':
        tasks = HCP_TASKS
    elif dataset == 'euskalibur':
        tasks = EUSKALIBUR_TASKS
    
    # preprocess selected tasks for each subject
    for subject in subjects:
        print(f'Starting preprocessing for subject: {subject}')
        pipeline = PreprocessingPipeline(dataset, subject)
        for task in tasks:
            print(f'Preprocessing task: {task} for subject: {subject}')
            pipeline.preprocess(task=task, save_physio_figs=True)
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform preprocessing pipeline on selected subject'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        required=True,
        default=None,
        choices=['hcp', 'euskalibur'],
        help='Dataset to perform preprocessing pipeline.',
    )
    parser.add_argument(
        '-s',
        '--subject',
        type=str,
        required=False,
        default=None,
        help='Subject to perform preprocessing pipeline. ' \
        'Only the subject ID is needed, e.g., sub-001. If not provided, ' \
        'the pipeline will be run for all subjects in the dataset.',
    )

    args = parser.parse_args()
    main(args.dataset, args.subject)