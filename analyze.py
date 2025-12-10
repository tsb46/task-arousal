"""
Perform full analysis pipeline on selected subject
"""

import argparse
import pickle

from typing import Literal

import nibabel as nib
import numpy as np

from task_arousal.analysis.pca import PCA
from task_arousal.analysis.complex_pca import ComplexPCA
from task_arousal.analysis.glm import (
    GLM,
    GLMPhysio
)
from task_arousal.analysis.dlm import (
    DistributedLagPhysioModel,
    DistributedLagEventModel,
    DistributedLagCommonalityAnalysis
)
from task_arousal.analysis.pls import (
    PLSEventPhysioModel
)
from task_arousal.analysis.rrr import (
    RRREventPhysioModel
)
from task_arousal.dataset.dataset_euskalibur import (
    DatasetEuskalibur, 
    PINEL_CONDITIONS,
    SIMON_CONDITIONS,
    MOTOR_CONDITIONS as MOTOR_CONDITIONS_EUSKALIBUR
)

from task_arousal.dataset.dataset_group import GroupDataset
from task_arousal.dataset.dataset_utils import DatasetLoad
from task_arousal.constants import (
    TR_EUSKALIBUR, 
    MASK_EUSKALIBUR,
)

# define output directory
OUT_DIRECTORY = 'results'

# physio signal labels
PHYSIO_LABELS_EUSKALIBUR = [
    'ppg_amplitude',
    'heart_rate',
    'resp_amp',
    'resp_rate',
    'endtidal_co2',
    'endtidal_o2',
]

# define all tasks (exclude Motor task)
TASKS_EUSKALIBUR = ['pinel', 'simon', 'motor', 'rest', 'breathhold']

# define tasks with event conditions
TASKS_EVENT_EUSKALIBUR = ['pinel', 'simon', 'motor']

# define analyses to perform
ANALYSES = [
    'glm_event', 
    'glm_physio', 
    'dlm_physio', 
    'dlm_event',
    'dlm_ca',
    'pca', 
    'cpca',
    'pls',
    'rrr',
]

# define Dataset type
Dataset = DatasetEuskalibur

def main(dataset: Literal['euskalibur'], subject: str | None, analysis: str | None) -> None:
    # initialize dataset loader
    if dataset == 'euskalibur':
        if subject is None:
            raise ValueError('Subject must be specified for EuskalIBUR dataset')
        ds = DatasetEuskalibur(subject=subject)
        TASKS = TASKS_EUSKALIBUR
        TASKS_EVENT = TASKS_EVENT_EUSKALIBUR
        TR = TR_EUSKALIBUR
        MASK = MASK_EUSKALIBUR
        PHYSIO_LABELS = PHYSIO_LABELS_EUSKALIBUR
        # For EUSKALIBUR, subject is guaranteed to be non-None
        _subject: str = subject
    else:
        raise NotImplementedError('Only EuskalIBUR dataset is currently supported')
    # if analysis is specified, only perform that analysis
    if analysis is not None:
        print(f'Performing only {analysis} analysis for subject {_subject}')
        _analysis = [analysis]
    else:
        _analysis = ANALYSES

    # only perform GLM analyses for tasks with event conditions
    if any(a in _analysis for a in ['glm_event']):
        for task in TASKS_EVENT:
            print(f'Loading data for subject {_subject}, task {task} for GLM analysis')
            # load nifti images without conversion to 2d array for input to nilearn GLM analysis
            data: DatasetLoad = ds.load_data(task=task, concatenate=False, convert_to_2d=False) # type: ignore
            # perform GLM analysis with event regressors
            _glm_event(dataset, data, ds, TR, MASK, _subject, task)
            print(f'GLM analyses complete for dataset {dataset}, subject {_subject}, task {task}')
    
    # perform PCA, Complex PCA, and DLM analyses for all tasks
    if any(a in _analysis for a in ['dlm_event', 'pca', 'cpca', 'dlm_physio']):
        for task in TASKS:
            print(f'Loading concatenated data for dataset {dataset}, subject {_subject}, task {task}')
            data: DatasetLoad = ds.load_data(task=task, concatenate=True) # type: ignore
                
            # perform PCA analysis
            if 'pca' in _analysis:
                _pca(dataset, data, ds, _subject, task)
                print(f'PCA analysis complete for dataset {dataset}, subject {_subject}, task {task}')
            # perform Complex PCA analysis
            if 'cpca' in _analysis:
                _cpca(dataset, data, ds, _subject, task)
                print(f'Complex PCA analysis complete for dataset {dataset}, subject {_subject}, task {task}')
            # perform DLM analysis with physiological regressors
            if 'dlm_physio' in _analysis:
                _dlm_physio(dataset, data, ds, TR, PHYSIO_LABELS, _subject, task)
                print(f'DLM with physiological regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}')

    # only perform DLM and GLM physio analyses for tasks with event conditions
    if any(a in _analysis for a in ['dlm_event', 'glm_physio', 'dlm_ca', 'pls', 'rrr']):
        for task in TASKS_EVENT:
            print(f'Loading data for dataset {dataset}, subject {_subject}, task {task} for DLM with event and GLM with physio analyses')
            data: DatasetLoad = ds.load_data(task=task, concatenate=False) # type: ignore

            if 'dlm_event' in _analysis:
                # perform DLM analysis with event regressors
                _dlm_event(dataset, data, ds, TR, _subject, task)
                print(f'DLM with event regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}')
            if 'glm_physio' in _analysis:
                # perform GLM analysis with physiological regressors
                _glm_physio(dataset, data, ds, TR, PHYSIO_LABELS, _subject, task)
                print(f'GLM with physiological regressors analysis complete for dataset {dataset}, subject {_subject}, task {task}')
            if 'dlm_ca' in _analysis:
                # perform commonality analysis with event and physiological regressors
                _dlm_ca(dataset, data, ds, TR, PHYSIO_LABELS, _subject, task)
                print(f'Commonality analysis complete for dataset {dataset}, subject {_subject}, task {task}')
            if 'pls' in _analysis:
                # perform PLS analysis with event and physiological regressors
                _pls(dataset, data, ds, TR, PHYSIO_LABELS, _subject, task)
                print(f'PLS analysis complete for dataset {dataset}, subject {_subject}, task {task}')
            if 'rrr' in _analysis:
                # perform RRR analysis with event and physiological regressors
                _rrr(dataset, data, ds, TR, PHYSIO_LABELS, _subject, task)
                print(f'RRR analysis complete for dataset {dataset}, subject {_subject}, task {task}')

def _cpca(
    dataset: str, 
    data: DatasetLoad, 
    ds: Dataset, 
    subject: str | None, 
    task: str
) -> None:
    """
    Perform Complex PCA analysis on the given data and save results to files.

    Parameters
    ----------
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing Complex PCA on subject {subject}, task {task}')
    # estimate Complex PCA with 10 components
    cpca = ComplexPCA(n_components=10)
    # run Complex PCA decomposition
    cpca_results = cpca.decompose(data['fmri'][0]) # type:ignore
    # write amplitude and phase of PC loadings to nifti file
    cpca_loadings_amp = ds.to_4d(cpca_results.loadings_amp.T)
    cpca_loadings_phase = ds.to_4d(cpca_results.loadings_phase.T)
    if dataset == "euskalibur":
        nib.nifti1.save(cpca_loadings_amp, f'{OUT_DIRECTORY}/sub-{subject}_{task}_cpca_loadings_amp.nii.gz')
        nib.nifti1.save(cpca_loadings_phase, f'{OUT_DIRECTORY}/sub-{subject}_{task}_cpca_loadings_phase.nii.gz')

        # write cpca metadata (including pc scores, exp var, etc.) to pickle file
        pickle.dump(
            cpca_results,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_cpca_metadata.pkl', 'wb')
        )

def _dlm_ca(
    dataset: str,
    data: DatasetLoad, 
    ds: Dataset, 
    tr: float,
    physio_labels: list[str],
    subject: str, 
    task: str
) -> None:
    """
    Perform commonality analysis on fMRI data, events and physio signals,
    and save results to files

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
    physio_labels : list[str]
        List of physiological regressor labels
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing commonality analysis on dataset {dataset}, subject {subject}, task {task}')
    # compute commonality analysis using DLM
    if dataset == 'euskalibur':
        physio_lags = 11

    dlm = DistributedLagCommonalityAnalysis(
        tr=tr,
        physio_lags=physio_lags,
        regressor_duration=25.0,
        n_knots_event=5,
        n_knots_physio=5,
    )
    dlm_res = dlm.fit(
        event_dfs=data['events'],
        fmri_data=data['fmri'], # type: ignore
        physio_data={
            physio_label: [d[physio_label].to_numpy()[:,np.newaxis] for d in data['physio']]
            for physio_label in physio_labels
        }
    )
    # write dlm metadata to pickle file
    if dataset == "euskalibur":
        pickle.dump(
            dlm_res,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_ca_metadata.pkl', 'wb')
        )


def _dlm_event(
    dataset: str, 
    data: DatasetLoad, 
    ds: Dataset, 
    tr: float,
    subject: str, 
    task: str
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
    print(f'Performing DLM with event regressors on dataset {dataset}, subject {subject}, task {task}')
    if dataset == 'euskalibur':
        if task  == 'pinel':
            conditions = PINEL_CONDITIONS
        elif task == 'simon':
            conditions = SIMON_CONDITIONS
        elif task == 'motor':
            conditions = MOTOR_CONDITIONS_EUSKALIBUR
        else:
            raise ValueError(f'Task {task} not recognized for DLM event analysis')
    
    dlm = DistributedLagEventModel(
        tr=tr, regressor_duration=20.0, n_knots=5, basis='cr'
    )
    dlm = dlm.fit(
        event_dfs=data['events'],
        outcome_data=data['fmri'] # type: ignore
    )
    # loop through conditions and write predicted functional time courses to nifti files
    for condition in conditions:
        dlm_eval = dlm.evaluate(trial=condition)
        pred_func_img = ds.to_4d(dlm_eval.pred_outcome)
        if dataset == "euskalibur":
            nib.nifti1.save(pred_func_img, f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_event_{condition}.nii.gz')
            # write dlm metadata (including betas, t-stats, etc.) to pickle file
            pickle.dump(
                dlm_eval,
                open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_event_{condition}_metadata.pkl', 'wb')
            )


def _dlm_physio(
    dataset: str,
    data: DatasetLoad, 
    ds: Dataset, 
    tr: float,
    physio_labels: list[str],
    subject: str, 
    task: str
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
    print(f'Performing DLM with physiological regressors on subject {subject}, task {task}')
    # loop through physio signals 
    for physio_label in physio_labels:
        # estimate DLM with physiological regressors
        if dataset == 'euskalibur':
            dlm = DistributedLagPhysioModel(
                tr=tr, neg_nlags=-3, nlags=8, n_knots=5, basis='cr'
            )

        dlm = dlm.fit(
            X=data['physio'][0][physio_label].to_numpy().reshape(-1,1),
            Y=data['fmri'][0] # type: ignore
        )
        # estimate functional time courses at each voxel to lagged physio signal
        dlm_eval = dlm.evaluate()
        # write predicted functional time courses to nifti file
        pred_func_img = ds.to_4d(dlm_eval.pred_outcome)
        if dataset == "euskalibur":
            nib.nifti1.save(pred_func_img, f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_physio_{physio_label}.nii.gz')

            # write dlm metadata to pickle file
            pickle.dump(
                dlm_eval,
                open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_physio_{physio_label}_metadata.pkl', 'wb')
            )


def _glm_event(
    dataset: str, 
    data: DatasetLoad, 
    ds: Dataset, 
    tr: float, 
    mask: str, 
    subject: str | None, 
    task: str
) -> None:
    """
    Perform General Linear Model (GLM) analysis with event regressors
    on the given data and save results to files. Perform analysis for both
    an 'spm' HRF model and a 'fir' HRF model.

    Parameters
    ----------
    dataset: str
        Dataset type ('euskalibur')
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    tr: float
        Repetition time (TR) of fMRI data
    mask : str
        Path to brain mask nifti file
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing GLM with event regressors on dataset {dataset}, subject {subject}, task {task}')
    # perform GLM analysis with 'spm' HRF model
    glm_spm = GLM(tr=tr, mask=mask, hrf='spm')
    glm_spm_results = glm_spm.fit(event_df=data['events'], fmri=data['fmri']) # type: ignore
    # write contrast maps to nifti files
    for glm_map_name, glm_map in glm_spm_results.contrast_maps.items():
        if dataset == "euskalibur":
            nib.nifti1.save(glm_map, f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_contrast_{glm_map_name}.nii.gz')

    # write glm metadata of HRF model (including contrasts, etc.) to pickle file
    if dataset == "euskalibur":
        pickle.dump(
            glm_spm_results,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_metadata.pkl', 'wb')
        )

    # perform GLM analysis with 'fir' HRF model
    if dataset == "euskalibur":
        fir_delays = np.arange(10)  # FIR with 10 TR delays for euskalibur

    glm_fir = GLM(tr=tr, mask=mask, hrf='fir', fir_delays=fir_delays)
    glm_fir_results = glm_fir.fit(event_df=data['events'], fmri=data['fmri']) # type: ignore
    # write contrast maps to nifti files
    for glm_map_name, glm_map in glm_fir_results.contrast_maps.items():
        if dataset == "euskalibur":
            nib.nifti1.save(glm_map, f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_fir_contrast_{glm_map_name}.nii.gz')

    # write glm metadata of FIR model (including contrasts, etc.) to pickle file
    if dataset == "euskalibur":
        pickle.dump(
            glm_fir_results,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_fir_metadata.pkl', 'wb')
        )


def _glm_physio(
    dataset: str, 
    data: DatasetLoad, 
    ds: Dataset,
    tr: float,
    physio_labels: list[str], 
    subject: str, 
    task: str
) -> None:
    """
    Perform General Linear Model (GLM) analysis with physiological regressors
    on the given data and save results to files. Analysis requires the choice
    of lag on physio regressor to account for hemodynamic delay. Default is 
    5.

    Parameters
    ----------
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    tr : float
        Repetition time (TR) of fMRI data
    physio_labels : list[str]
        List of physiological regressor labels
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing GLM with physiological regressors on dataset {dataset}, subject {subject}, task {task}')
    
    if dataset == 'euskalibur':
        if task  == 'pinel':
            conditions = PINEL_CONDITIONS
        elif task == 'simon':
            conditions = SIMON_CONDITIONS
        elif task == 'motor':
            conditions = MOTOR_CONDITIONS_EUSKALIBUR
        else:
            raise ValueError(f'Task {task} not recognized for GLM physio analysis')
    
    # loop through physio signals
    for physio_label in physio_labels:
        print(f'Performing GLM with physiological regressor {physio_label}')
        # perform GLM physio analysis with 'spm' HRF model
        if dataset == 'euskalibur':
            glm_physio = GLMPhysio(tr, hrf='spm', physio_lag=5)

        glm_physio.fit(
            event_df=data['events'],
            fmri=data['fmri'], # type: ignore
            physio=[d[physio_label].to_numpy()[:,np.newaxis] for d in data['physio']]
        )
        # write contrast maps to nifti files
        for contrast_name in conditions:
            glm_eval = glm_physio.evaluate(contrast_name)
            pred_func_img = ds.to_4d(glm_eval.pred_func)
            if dataset == "euskalibur":
                nib.nifti1.save( # type: ignore
                    pred_func_img, 
                    f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_{physio_label}_contrast_{contrast_name}.nii.gz'
                )
                # write glm metadata of HRF model (including contrasts, etc.) to pickle file
                pickle.dump(
                    glm_eval,
                    open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_{physio_label}_contrast_{contrast_name}_metadata.pkl', 'wb')
                )


def _pca(
    dataset: str,
    data: DatasetLoad, 
    ds: Dataset, 
    subject: str | None, 
    task: str
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
    print(f'Performing PCA on dataset {dataset}, subject {subject}, task {task}')
    # estimate PCA with 10 components
    pca = PCA(n_components=10)
    # run PCA decomposition
    pca_results = pca.decompose(data['fmri'][0])
    # write loadings to nifti file
    pca_loadings = ds.to_4d(pca_results.loadings.T)
    if dataset == "euskalibur":
        nib.nifti1.save(pca_loadings, f'{OUT_DIRECTORY}/sub-{subject}_{task}_pca_loadings.nii.gz')
        # write pca metadata (including pc scores, exp var, etc.) to pickle file
        pickle.dump(
            pca_results,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_pca_metadata.pkl', 'wb')
        )



def _pls(
    dataset: str, 
    data: DatasetLoad, 
    ds: Dataset, 
    tr: float,
    physio_labels: list[str],
    subject: str, 
    task: str
) -> None:
    """
    Perform PLS decomposition on fMRI data, events and physio signals,
    and save results to files

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
    print(f'Performing PLS on dataset {dataset}, subject {subject}, task {task}')
    if dataset == 'euskalibur':
        physio_lags = 11

    # estimate PLS with 10 components
    pls = PLSEventPhysioModel(
        tr=tr,
        n_components=10,
        physio_lags=physio_lags,
        regressor_duration=25.0,
        n_knots_event=5,
        n_knots_physio=5,
    )    
    pls_res = pls.fit(
        event_dfs=data['events'],
        fmri_data=data['fmri'], # type: ignore
        physio_data={
            physio_label: [d[physio_label].to_numpy()[:,np.newaxis] for d in data['physio']]
            for physio_label in physio_labels
        }
    )
    # write pls metadata to pickle file
    if dataset == "euskalibur":
        pickle.dump(
            pls_res,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_pls_metadata.pkl', 'wb')
        )

def _rrr(
    dataset: str,
    data: DatasetLoad, 
    ds: Dataset, 
    tr: float,
    physio_labels: list[str],
    subject: str, 
    task: str
) -> None:
    """
    Perform Reduced Rank Regression on fMRI data, events and physio signals,
    and save results to files

    Parameters
    ----------
    dataset : str
        Dataset identifier
    data : DatasetLoad
        Loaded dataset containing fMRI data and associated information
    ds : Dataset
        Dataset object for handling data operations
    tr : float
        Repetition time of the fMRI data
    physio_labels : list[str]
        List of physiological signal labels
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing RRR on subject dataset {dataset}, subject {subject}, task {task}')
    if dataset == 'euskalibur':
        physio_lags = 11

    # estimate RRR with 10 components
    rrr = RRREventPhysioModel(
        tr=tr,
        n_components=10,
        physio_lags=physio_lags,
        regressor_duration=25.0,
        n_knots_event=5,
        n_knots_physio=5,
    )
    rrr_res = rrr.fit(
        event_dfs=data['events'],
        fmri_data=data['fmri'], # type: ignore
        physio_data={
            physio_label: [d[physio_label].to_numpy()[:,np.newaxis] for d in data['physio']]
            for physio_label in physio_labels
        }
    )
    # write rrr metadata to pickle file
    if dataset == "euskalibur":
        pickle.dump(
            rrr_res,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_rrr_metadata.pkl', 'wb')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform full analysis pipeline on selected subject'
    )
    # add dataset argument
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        required=False,
        default='euskalibur',
        choices=['euskalibur'],
        help='Dataset to perform analysis pipeline on',
    )
    # add subject argument
    parser.add_argument(
        '-s',
        '--subject',
        type=str,
        default=None,
        required=False,
        help='Subject to perform analysis pipeline. Required if dataset is euskalibur.',
    )
    # add optional analysis argument (default: all analyses)
    parser.add_argument(
        '-a',
        '--analysis',
        type=str,
        choices=ANALYSES,
        required=False,
        default=None,
        help='Type of analysis to perform'
    )
    # parse arguments
    args = parser.parse_args()
    main(args.dataset, args.subject, args.analysis)