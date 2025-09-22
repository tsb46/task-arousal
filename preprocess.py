"""
Perform full preprocessing pipeline on selected subject
"""

import argparse

from task_arousal.preprocess.pipeline import PreprocessingPipeline

# define tasks to preprocess (exclude Motor task)
TASKS = ['pinel', 'simon', 'rest', 'breathhold']

def main(subject):
    pipeline = PreprocessingPipeline(subject)
    # loop through tasks and preprocess
    for task in TASKS:
        print(f'Preprocessing task: {task} for subject: {subject}')
        pipeline.preprocess(task=task, save_physio_figs=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform preprocessing pipeline on selected subject'
    )
    parser.add_argument(
        '-s',
        '--subject',
        type=str,
        required=True,
        help='Subject to perform preprocessing pipeline',
    )

    args = parser.parse_args()
    main(args.subject)