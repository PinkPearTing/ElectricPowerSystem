import numpy as np

def fitcalcPRE(s, SERA, SERC, SERD, SERE):
    """
    Calculate fitted Y values based on the provided parameters.

    Parameters:
        s: array-like, input values
        SERA: array-like, the A matrix
        SERC: array-like, the C matrix
        SERD: array-like, the D matrix
        SERE: array-like, the E matrix

    Returns:
        Yfit: 3D array of calculated fitted values
    """

    Ns = len(s)
    Nc = len(SERD)
    N = len(SERA)
    Yfit = np.zeros((Nc, Nc, Ns))

    for k in range(Ns):
        Y = np.zeros((Nc, Nc))
        for row in range(Nc):
            for col in range(Nc):
                Y[row, col] = SERD[row, col] + s[k] * SERE[row, col]
                Y[row, col] += np.sum(SERC[row, col, :N] / (s[k] - SERA[:N]))

        Yfit[:, :, k] = Y

    return Yfit