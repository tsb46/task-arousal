"""
Group-level dataset orchestrator that loads multiple subjects for a given dataset/task.
"""

from __future__ import annotations

from typing import Callable, List, Literal, Protocol, Any, Iterator, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

from task_arousal.io.file import get_dataset_subjects
from .dataset_euskalibur import DatasetEuskalibur
from task_arousal.constants import MASK_EUSKALIBUR, MASK_PAN
from .dataset_utils import to_img as _to_img, DatasetLoad


# Protocol for subject loader classes
class SubjectLoader(Protocol):
    def load_data(self, task: str, **kwargs) -> Any: ...


class GroupDataset:
    """
    Load all subjects for a task from a dataset using the appropriate subject loader.
    """

    def __init__(
        self,
        dataset: Literal["euskalibur"],
        subjects: List[str] | None = None,
        subject_loader_factory: Callable[[str], SubjectLoader] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the GroupDataset object.

        Parameters
        ----------
        dataset : {'euskalibur'}
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
            if dataset == "euskalibur":
                self._factory = lambda s: DatasetEuskalibur(s)
            else:
                raise ValueError("Unsupported dataset: must be 'euskalibur'")

    def load_data(
        self,
        task: str,
        # Group-level controls that are forwarded to subject loaders
        concatenate: bool = True,
        normalize: bool = True,
        load_func: bool = True,
        load_physio: bool = True,
        # Orchestration behavior
        stream: bool = False,
        on_error: Literal["skip", "raise"] = "raise",
        **kwargs,
    ) -> Union[Iterator[Tuple[str, Any]], DatasetLoad]:
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
                            load_func=load_func,
                            load_physio=load_physio,
                            verbose=self.verbose,
                            **kwargs,
                        )
                        yield subj, result
                    except Exception as e:
                        if self.verbose:
                            print(f"Subject {subj} failed: {e}")
                        if on_error == "raise":
                            raise
                        continue

            return _stream()
        else:
            out_lists: dict[str, list] = {"fmri": [], "physio": [], "events": []}
            for subj in self.subjects:
                try:
                    loader = self._factory(subj)
                    result = loader.load_data(
                        task=task,
                        concatenate=concatenate,
                        normalize=normalize,
                        load_func=load_func,
                        load_physio=load_physio,
                        verbose=self.verbose,
                        **kwargs,
                    )
                    # If results are concatenated within subject, take first element
                    out_lists["fmri"].append(
                        result["fmri"][0] if concatenate else result["fmri"]
                    )
                    out_lists["physio"].append(
                        result["physio"][0] if concatenate else result["physio"]
                    )
                    out_lists["events"].append(
                        result["events"][0] if concatenate else result["events"]
                    )

                except Exception as e:
                    if self.verbose:
                        print(f"Subject {subj} failed: {e}")
                    if on_error == "raise":
                        raise
                    continue
            # if specified, concatenate across subjects
            if concatenate:
                if self.verbose:
                    print("Concatenating data across subjects...")
                out: DatasetLoad = self._post_concatenate_subjects(out_lists)
            else:
                out = out_lists  # type: ignore
            return out

    def _post_concatenate_subjects(self, dataset_load: dict[str, list]) -> DatasetLoad:
        """
        Concatenate data across subjects after loading.
        Expects dataset_load to have keys 'fmri' (List[np.ndarray or Nifti]) and 'physio' (List[pd.DataFrame]).
        """

        # Concatenate physio tables
        physio_concat = pd.concat(dataset_load["physio"], ignore_index=True)

        # Temporal concatenation fmri
        fmri_concat = np.concatenate(dataset_load["fmri"], axis=0)

        return {
            "fmri": fmri_concat,
            "physio": physio_concat,
            "events": dataset_load["events"],
        }

    def to_img(self, fmri_data: np.ndarray) -> nib.Nifti1Image:  # type: ignore
        """
        Convert time x voxels array back to a 4D NIfTI image via shared utils.
        """
        if self.dataset == "euskalibur":
            return _to_img(fmri_data, MASK_EUSKALIBUR)  # type: ignore
        elif self.dataset == "pan":
            return _to_img(fmri_data, MASK_PAN)  # type: ignore
        else:
            raise ValueError("Unsupported dataset for to_img conversion.")
        return _to_img(fmri_data, self.mask)  # type: ignore
