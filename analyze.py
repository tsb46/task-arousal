"""
Perform full analysis pipeline on selected subject
"""

import argparse
import pickle

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
    DistributedLagEventModel
)
from task_arousal.analysis.pls import (
    PLSEventPhysioModel
)
from task_arousal.analysis.dataset import (
    Dataset, 
    DatasetLoad,
    PINEL_CONDITIONS,
    SIMON_CONDITIONS
)

# define output directory
OUT_DIRECTORY = 'results'

# physio signal labels
PHYSIO_LABELS = [
    'ppg_amplitude',
    'heart_rate',
    'resp_amp',
    'resp_rate',
    'endtidal_co2',
    'endtidal_o2',
]
# define all tasks (exclude Motor task)
TASKS = ['pinel', 'simon', 'rest', 'breathhold']

# define tasks with event conditions
TASKS_EVENT = ['pinel', 'simon']

# define analyses to perform
ANALYSES = [
    'glm_event', 
    'glm_physio', 
    'dlm_physio', 
    'dlm_event', 
    'pca', 
    'cpca',
    'pls'
]

def main(subject: str, analysis: str | None) -> None:
    # initialize dataset loader
    ds = Dataset(subject=subject)
    # if analysis is specified, only perform that analysis
    if analysis is not None:
        print(f'Performing only {analysis} analysis for subject {subject}')
        _analysis = [analysis]
    else:
        _analysis = ANALYSES

    # only perform GLM analyses for tasks with event conditions
    if any(a in _analysis for a in ['glm_event']):
        for task in TASKS_EVENT:
            print(f'Loading data for subject {subject}, task {task} for GLM analysis')
            # load nifti images without conversion to 2d array for input to nilearn GLM analysis
            data = ds.load_data(task=task, concatenate=False, convert_to_2d=False)
            # perform GLM analysis with event regressors
            _glm_event(data, ds, subject, task)
            print(f'GLM analyses complete for subject {subject}, task {task}')
    
    # perform PCA, Complex PCA, and DLM analyses for all tasks
    if any(a in _analysis for a in ['dlm_event', 'pca', 'cpca', 'dlm_physio']):
        for task in TASKS:
            print(f'Loading concatenated data for subject {subject}, task {task}')
            data = ds.load_data(task=task, concatenate=True)
            # perform PCA analysis
            if 'pca' in _analysis:
                _pca(data, ds, subject, task)
                print(f'PCA analysis complete for subject {subject}, task {task}')
            # perform Complex PCA analysis
            if 'cpca' in _analysis:
                _cpca(data, ds, subject, task)
                print(f'Complex PCA analysis complete for subject {subject}, task {task}')
            # perform DLM analysis with physiological regressors
            if 'dlm_physio' in _analysis:
                _dlm_physio(data, ds, subject, task)
                print(f'DLM with physiological regressors analysis complete for subject {subject}, task {task}')

    # only perform DLM and GLM physio analyses for tasks with event conditions
    if any(a in _analysis for a in ['dlm_event', 'glm_physio', 'pls']):
        for task in TASKS_EVENT:
            print(f'Loading data for subject {subject}, task {task} for DLM with event and GLM with physio analyses')
            data = ds.load_data(task=task, concatenate=False)
            if 'dlm_event' in _analysis:
                # perform DLM analysis with event regressors
                _dlm_event(data, ds, subject, task)
                print(f'DLM with event regressors analysis complete for subject {subject}, task {task}')
            if 'glm_physio' in _analysis:
                # perform GLM analysis with physiological regressors
                _glm_physio(data, ds, subject, task)
                print(f'GLM with physiological regressors analysis complete for subject {subject}, task {task}')
            if 'pls' in _analysis:
                # perform PLS analysis with event and physiological regressors
                _pls(ds, subject, task)
                print(f'PLS analysis complete for subject {subject}, task {task}')
    
    

def _cpca(data: DatasetLoad, ds: Dataset, subject: str, task: str) -> None:
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
    nib.save(cpca_loadings_amp, f'{OUT_DIRECTORY}/sub-{subject}_{task}_cpca_loadings_amp.nii.gz') # type:ignore
    nib.save(cpca_loadings_phase, f'{OUT_DIRECTORY}/sub-{subject}_{task}_cpca_loadings_phase.nii.gz') # type:ignore
    # write cpca metadata (including pc scores, exp var, etc.) to pickle file
    pickle.dump(
        cpca_results,
        open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_cpca_metadata.pkl', 'wb')
    )


def _dlm_event(data: DatasetLoad, ds: Dataset, subject: str, task: str) -> None:
    """
    Perform Distributed Lag Model (DLM) analysis with event regressors
    on the given data and save results to files.

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
    print(f'Performing DLM with event regressors on subject {subject}, task {task}')
    if task  == 'pinel':
        conditions = PINEL_CONDITIONS
    elif task == 'simon':
        conditions = SIMON_CONDITIONS
    else:
        raise ValueError(f'Task {task} not recognized for DLM event analysis')
    
    dlm = DistributedLagEventModel(
        regressor_duration=20.0, n_knots=5, basis='cr'
    )
    dlm = dlm.fit(
        event_dfs=data['events'],
        outcome_data=data['fmri'] # type: ignore
    )
    # loop through conditions and write predicted functional time courses to nifti files
    for condition in conditions:
        dlm_eval = dlm.evaluate(trial=condition)
        pred_func_img = ds.to_4d(dlm_eval.pred_outcome)
        nib.save(pred_func_img, f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_event_{condition}.nii.gz') # type:ignore

        # write dlm metadata (including betas, t-stats, etc.) to pickle file
        pickle.dump(
            dlm_eval,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_event_{condition}_metadata.pkl', 'wb')
        )


def _dlm_physio(data: DatasetLoad, ds: Dataset, subject: str, task: str) -> None:
    """
    Perform Distributed Lag Model (DLM) analysis with physiological regressors
    on the given data and save results to files.

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
    print(f'Performing DLM with physiological regressors on subject {subject}, task {task}')
    # loop through physio signals 
    for physio_label in PHYSIO_LABELS:
        # estimate DLM with physiological regressors - using lags -3 to +8 TRs
        dlm = DistributedLagPhysioModel(
            neg_nlags=-3, nlags=8, n_knots=5, basis='cr'
        )
        dlm = dlm.fit(
            X=data['physio'][0][physio_label].to_numpy().reshape(-1,1),
            Y=data['fmri'][0] # type: ignore
        )
        # estimate functional time courses at each voxel to lagged physio signal
        dlm_eval = dlm.evaluate()
        # write predicted functional time courses to nifti file
        pred_func_img = ds.to_4d(dlm_eval.pred_outcome)
        nib.save(pred_func_img, f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_physio_{physio_label}.nii.gz') # type:ignore

        # write dlm metadata (including betas, t-stats, etc.) to pickle file
        pickle.dump(
            dlm_eval,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_dlm_physio_{physio_label}_metadata.pkl', 'wb')
        )


def _glm_event(data: DatasetLoad, ds: Dataset, subject: str, task: str) -> None:
    """
    Perform General Linear Model (GLM) analysis with event regressors
    on the given data and save results to files. Perform analysis for both
    an 'spm' HRF model and a 'fir' HRF model.

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
    print(f'Performing GLM with event regressors on subject {subject}, task {task}')
    # perform GLM analysis with 'spm' HRF model
    glm_spm = GLM(hrf='spm')
    glm_spm_results = glm_spm.fit(event_df=data['events'], fmri=data['fmri']) # type: ignore
    # write contrast maps to nifti files
    for glm_map_name, glm_map in glm_spm_results.contrast_maps.items():
        nib.save(glm_map, f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_contrast_{glm_map_name}.nii.gz') # type:ignore
    # write glm metadata of HRF model (including contrasts, etc.) to pickle file
    pickle.dump(
        glm_spm_results,
        open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_metadata.pkl', 'wb')
    )

    # perform GLM analysis with 'fir' HRF model
    glm_fir = GLM(hrf='fir', fir_delays=np.arange(10)) # FIR with 10 TR delays
    glm_fir_results = glm_fir.fit(event_df=data['events'], fmri=data['fmri']) # type: ignore
    # write contrast maps to nifti files
    for glm_map_name, glm_map in glm_fir_results.contrast_maps.items():
        nib.save(glm_map, f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_fir_contrast_{glm_map_name}.nii.gz') # type:ignore
    # write glm metadata of FIR model (including contrasts, etc.) to pickle file
    pickle.dump(
        glm_fir_results,
        open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_fir_metadata.pkl', 'wb')
    )


def _glm_physio(data: DatasetLoad, ds: Dataset, subject: str, task: str) -> None:
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
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing GLM with physiological regressors on subject {subject}, task {task}')
    
    if task  == 'pinel':
        conditions = PINEL_CONDITIONS
    elif task == 'simon':
        conditions = SIMON_CONDITIONS
    else:
        raise ValueError(f'Task {task} not recognized for GLM physio analysis')
    
    # loop through physio signals
    for physio_label in PHYSIO_LABELS:
        print(f'Performing GLM with physiological regressor {physio_label}')
        # perform GLM physio analysis with 'spm' HRF model
        glm_physio = GLMPhysio(hrf='spm', physio_lag=5)
        glm_physio.fit(
            event_df=data['events'],
            fmri=data['fmri'], # type: ignore
            physio=[d[physio_label].to_numpy()[:,np.newaxis] for d in data['physio']]
        )
        # write contrast maps to nifti files
        for contrast_name in conditions:
            glm_eval = glm_physio.evaluate(contrast_name)
            pred_func_img = ds.to_4d(glm_eval.pred_func)
            nib.save( # type: ignore
                pred_func_img, 
                f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_{physio_label}_contrast_{contrast_name}.nii.gz'
            )
            # write glm metadata of HRF model (including contrasts, etc.) to pickle file
            pickle.dump(
                glm_eval,
                open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_glm_hrf_{physio_label}_contrast_{contrast_name}_metadata.pkl', 'wb')
            )


def _pca(data: DatasetLoad, ds: Dataset, subject: str, task: str) -> None:
    """
    Perform PCA decomposition on fMRI data and save results to files

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
    print(f'Performing PCA on subject {subject}, task {task}')
    # estimate PCA with 10 components
    pca = PCA(n_components=10)
    # run PCA decomposition
    pca_results = pca.decompose(data['fmri'][0]) # type:ignore
    # write loadings to nifti file
    pca_loadings = ds.to_4d(pca_results.loadings.T)
    nib.save(pca_loadings, f'{OUT_DIRECTORY}/sub-{subject}_{task}_pca_loadings.nii.gz') # type:ignore
    # write pca metadata (including pc scores, exp var, etc.) to pickle file
    pickle.dump(
        pca_results,
        open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_pca_metadata.pkl', 'wb')
    )

def _pls(ds: Dataset, subject: str, task: str) -> None:
    """
    Perform PLS decomposition on fMRI data, events and physio signals,
    and save results to files

    Parameters
    ----------
    ds : Dataset
        Dataset object for handling data operations
    subject : str
        Subject identifier
    task : str
        Task identifier
    """
    print(f'Performing PLS on subject {subject}, task {task}')
    # load concatenated data
    data = ds.load_data(task=task, concatenate=True)
    # estimate PLS for each physio signal
    for physio_label in PHYSIO_LABELS:
        print(f'Performing PLS with physiological regressor {physio_label}')
        # estimate PLS with 10 components
        pls = PLSEventPhysioModel(
            n_components=10,
            physio_lags=11,
            regressor_duration=25.0,
            n_knots_event=5,
            n_knots_physio=5,
        )    
        pls_res = pls.fit(
            event_dfs=data['events'],
            fmri_data=data['fmri'], # type: ignore
            physio_data=[d[physio_label].to_numpy()[:,np.newaxis] for d in data['physio']]
        )
        # write pls metadata (including pc scores, exp var, etc.) to pickle file
        pickle.dump(
            pls_res,
            open(f'{OUT_DIRECTORY}/sub-{subject}_{task}_{physio_label}_pls_metadata.pkl', 'wb')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform full analysis pipeline on selected subject'
    )
    # add subject argument
    parser.add_argument(
        '-s',
        '--subject',
        type=str,
        required=True,
        help='Subject to perform analysis pipeline',
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
    main(args.subject, args.analysis)