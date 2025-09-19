"""
Utility functions for analysis module
"""

import numpy as np


def create_interaction_matrix(
    event_regs: np.ndarray,
    physio_reg: np.ndarray
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
        (n_time, n_event_regs * n_physio_regs),
        dtype=event_regs.dtype
    )
    for i in range(n_event_regs):
        for j in range(n_physio_regs):
            interaction_mat[:, i*n_physio_regs + j] = (
                event_regs[:, i] * physio_reg[:, j]
            )
    
    return interaction_mat


def lag_mat(x: np.ndarray, lags: list[int], fill_val: float = np.nan) -> np.ndarray:
    """
    Create array of time-lagged copies of the time course. Modified
    for negative lags from:
    https://github.com/ulf1/lagmat

    Parameters
    ----------
        x : np.ndarray
            The time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        lags : list[int]
            List of integer lags (shifts) to apply to the time course.
            Positive values indicate a lag (shift down), negative values
            indicate a lead (shift up).
        fill_val : float, optional
            Value to use for filling in missing values after shifting.
            Defaults to np.nan.

    """
    n_rows, n_cols = x.shape
    n_lags = len(lags)
    # allocate memory
    x_lag = np.empty(
        shape=(n_rows, n_cols * n_lags),
        order='F', dtype=x.dtype
    )
    # fill w/ Nans
    x_lag[:] = fill_val
    # Copy lagged columns of X into X_lag
    for i, l in enumerate(lags):
        # target columns of X_lag
        j = i * n_cols
        k = j + n_cols  # (i+1) * ncols
        # number rows of X
        nl = n_rows - abs(l)
        # Copy
        if l >= 0:
            x_lag[l:, j:k] = x[:nl, :]
        else:
            x_lag[:l, j:k] = x[-nl:, :]
    return x_lag
