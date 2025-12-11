"""
Dataset module for loading preprocessed data for HCP and Euskalibur datasets.
"""

from .dataset_euskalibur import DatasetEuskalibur
from .dataset_group import GroupDataset

all = [
    'DatasetEuskalibur',
    'GroupDataset'
]


