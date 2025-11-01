"""
Group-level dataset orchestrator that loads multiple subjects for a given dataset/task.
"""
from __future__ import annotations

from typing import Callable, List, Literal, Protocol, Any

import nibabel as nib 
import numpy as np
import pandas as pd

from task_arousal.io.file import get_dataset_subjects
from .dataset_hcp import DatasetHCPSubject
from .dataset_euskalibur import DatasetEuskalibur
from task_arousal.constants import MASK_HCP, MASK_EUSKALIBUR
from .dataset_utils import to_4d as _to_4d


# Protocol for subject loader classes
class SubjectLoader(Protocol):
    def load_data(self, task: str, **kwargs) -> Any:
        ...


class GroupDataset:
    """
    Load all subjects for a task from a dataset using the appropriate subject loader.
    """

    def __init__(
        self,
        dataset: Literal['euskalibur', 'hcp'],
        subjects: List[str] | None = None,
        subject_loader_factory: Callable[[str], SubjectLoader] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the GroupDataset object.

        Parameters
        ----------
        dataset : {'euskalibur', 'hcp'}
            Dataset name.
        subjects : List[str] | None
            List of subject IDs to load. If None, loads all subjects in the dataset.
        subject_loader_factory : Callable[[str], SubjectLoader] | None
            Factory function to create subject loader instances. If None, uses default loaders.
        verbose : bool
            If True, prints progress and error messages.
        """
        self.dataset = dataset
        self.verbose = verbose

        if subjects is None:
            self.subjects = get_dataset_subjects(dataset)
        else:
            self.subjects = [str(s) for s in subjects]
        # factory to build a subject loader
        if subject_loader_factory is not None:
            self._factory = subject_loader_factory
        else:
            if dataset == 'hcp':
                self._factory = lambda s: DatasetHCPSubject(s)
            elif dataset == 'euskalibur':
                self._factory = lambda s: DatasetEuskalibur(s)
            else:
                raise ValueError("Unsupported dataset: must be 'euskalibur' or 'hcp'")

    def load_data(
        self,
        task: str,
        # Group-level controls that are forwarded to subject loaders
        concatenate: bool = True,
        normalize: bool = True,
        convert_to_2d: bool = True,
        load_func: bool = True,
        load_physio: bool = True,
        # Orchestration behavior
        stream: bool = False,
        on_error: Literal['skip', 'raise'] = 'raise',
        **kwargs,
    ):
        """
        Load data for all configured subjects.

        Parameters
        ----------
        task : str
            Task identifier.
        concatenate : bool
            If True, concatenate runs within and across each subject.
        normalize : bool
            If True, z-score normalize the fMRI data within each subject before concatenation.
        convert_to_2d : bool
            If True, convert fMRI data to 2D (time x voxels).
        load_func : bool
            If True, load fMRI data.
        load_physio : bool
            If True, load physiological data.
        stream : bool
            If True, yields (subject, DatasetLoad) tuples; otherwise returns a dict. Concatenation
            across subjects is not performed when streaming.
        on_error : {'skip','raise'}
            Skip or raise on per-subject failures.
        kwargs :
            Forwarded to subject.load_data(...).
        """
        if stream:
            def _stream():
                for subj in self.subjects:
                    try:
                        loader = self._factory(subj)
                        result = loader.load_data(
                            task=task,
                            concatenate=concatenate,
                            normalize=normalize,
                            convert_to_2d=convert_to_2d,
                            load_func=load_func,
                            load_physio=load_physio,
                            verbose=self.verbose,
                            **kwargs,
                        )
                        yield subj, result
                    except Exception as e:
                        if self.verbose:
                            print(f"Subject {subj} failed: {e}")
                        if on_error == 'raise':
                            raise
                        continue
            return _stream()
        else:
            out = {
                "fmri": [],
                "physio": [],
                "events": []
            }
            for subj in self.subjects:
                try:
                    loader = self._factory(subj)
                    result = loader.load_data(
                        task=task,
                        concatenate=concatenate,
                        normalize=normalize,
                        convert_to_2d=convert_to_2d,
                        load_func=load_func,
                        load_physio=load_physio,
                        verbose=self.verbose,
                        **kwargs,
                    )
                    # If results are concatenated within subject, take first element
                    out['fmri'].append(result['fmri'][0] if concatenate else result['fmri'])
                    out['physio'].append(result['physio'][0] if concatenate else result['physio'])
                    out['events'].append(result['events'][0] if concatenate else result['events'])

                except Exception as e:
                    if self.verbose:
                        print(f"Subject {subj} failed: {e}")
                    if on_error == 'raise':
                        raise
                    continue
            # if specified, concatenate across subjects
            if concatenate:
                if self.verbose:
                    print("Concatenating data across subjects...")
                out = self._post_concatenate_subjects(out, convert_to_2d)
            return out

    def _post_concatenate_subjects(self, dataset_load: dict[str, Any], convert_to_2d: bool) -> Any:
        """
        Concatenate data across subjects after loading.
        Expects dataset_load to have keys 'fmri' (List[np.ndarray or Nifti]) and 'physio' (List[pd.DataFrame]).
        """

        # Concatenate physio tables
        dataset_load['physio'] = pd.concat(dataset_load['physio'], ignore_index=True)

        # Concatenate fmri if 2D
        if convert_to_2d:
            dataset_load['fmri'] = np.concatenate(dataset_load['fmri'], axis=0)
        else:
            # cannot concatenate 4D here; leave unchanged
            print("Warning: skipping group concatenation of non-2D data.")
            dataset_load['fmri'] = dataset_load['fmri']
        
        return dataset_load
    
    def to_4d(
        self,
        fmri_data: np.ndarray
    ) -> nib.Nifti1Image: # type: ignore
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.
        """
        if self.dataset == 'hcp':
            return _to_4d(fmri_data, MASK_HCP) # type: ignore
        elif self.dataset == 'euskalibur':
            return _to_4d(fmri_data, MASK_EUSKALIBUR) # type: ignore
        return _to_4d(fmri_data, self.mask) # type: ignore