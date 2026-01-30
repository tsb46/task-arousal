"""
Utility functions for analysis module
"""

import numpy as np


def create_interaction_matrix(
    event_regs: np.ndarray, physio_reg: np.ndarray
) -> np.ndarray:
    """
    create interaction matrix between event regressors and physio regressor

    Parameters
    ----------
    event_regs: np.ndarray
        Event regressors (2D - time x event regressors)
    physio_reg: np.ndarray
        Physio regressor (2D - time x physio regressors)

    Returns
    -------
    interaction_mat: np.ndarray
        Interaction matrix between event and physio regressors
        (2D - time x (event regressors * physio regressors))
    """
    n_time = event_regs.shape[0]
    n_event_regs = event_regs.shape[1]
    n_physio_regs = physio_reg.shape[1]
    # allocate memory for interaction matrix
    interaction_mat = np.empty(
        (n_time, n_event_regs * n_physio_regs), dtype=event_regs.dtype
    )
    for i in range(n_event_regs):
        for j in range(n_physio_regs):
            interaction_mat[:, i * n_physio_regs + j] = (
                event_regs[:, i] * physio_reg[:, j]
            )

    return interaction_mat
