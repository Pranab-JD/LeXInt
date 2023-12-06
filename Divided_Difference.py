import numpy as np

def Divided_Difference(X, diffs):
    """
    Parameters
    ----------
    X       : Leja points
    diffs   : Phi function array

    Returns
    -------
    div_diff : Polynomial coefficients

    """

    N = len(X)
    div_diff = diffs
    
    for ii in range(1, N):
        div_diff[ii:N] = (div_diff[ii:N] - div_diff[ii - 1])/(X[ii:N] - X[ii - 1])

    return div_diff
