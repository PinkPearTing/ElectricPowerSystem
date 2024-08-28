import numpy as np


def fitcalcABCDE(sk, A, B, C, D, E):
    """
    Calculate the fitted Y values.

    Parameters:
        sk: array-like, input values
        A: array-like, the A matrix
        B: array-like, the B matrix
        C: array-like, the C matrix
        D: array-like, the D matrix
        E: array-like, the E matrix

    Returns:
        Yfit: array-like, calculated fitted values
    """

    Nc = len(D)
    N = len(A)

    # Create a matrix to perform the element-wise division
    dum = np.repmat(1 / (sk - A), Nc, 1).T  # Transpose to match shape

    # Update C by element-wise multiplication
    C_dum = C * dum

    # Calculate Yfit
    Yfit = C_dum @ B + D + sk * E

    return Yfit