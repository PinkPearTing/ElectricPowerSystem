import numpy as np


def intercheig(V, oldV, D, Nc, fstep):
    """
    Interchange eigenvectors and corresponding eigenvalues to ensure smooth functions of frequency.

    Parameters:
        V: ndarray, new eigenvectors
        oldV: ndarray, old eigenvectors
        D: ndarray, associated eigenvalues
        Nc: int, number of columns (or eigenvectors)
        fstep: int, frequency step

    Returns:
        V: ndarray, updated eigenvectors
        D: ndarray, updated eigenvalues
    """

    if fstep > 1:
        UGH = np.abs(np.real(oldV.T @ V))
        dot = np.zeros(Nc)
        ind = np.arange(Nc)
        taken = np.zeros(Nc, dtype=int)

        # Find largest dot products
        for ii in range(Nc):
            ilargest = 0
            rlargest = 0
            for j in range(Nc):
                dotprod = UGH[ii, j]
                if dotprod > rlargest:
                    rlargest = np.abs(np.real(dotprod))
                    ilargest = j
            dot[ii] = rlargest
            ind[ii] = ii
            taken[ii] = 0

        # Sort indices based on dot products in descending order
        ind = ind[np.argsort(-dot)]

        # Perform interchanges in prioritized sequence
        for l in range(Nc):
            ii = ind[l]
            ilargest = 0
            rlargest = 0

            for j in range(Nc):
                if taken[j] == 0:
                    dotprod = UGH[ii, j]
                    if dotprod > rlargest:
                        rlargest = np.abs(np.real(dotprod))
                        ilargest = j

            taken[ii] = 1

            # Swap eigenvectors and corresponding eigenvalues
            V[:, [ii, ilargest]] = V[:, [ilargest, ii]]
            D[ii, ii], D[ilargest, ilargest] = D[ilargest, ilargest], D[ii, ii]

            # Swap columns in UGH
            UGH[:, [ii, ilargest]] = UGH[:, [ilargest, ii]]

        # Adjust signs of eigenvectors if needed
        for ii in range(Nc):
            dotprod = oldV[:, ii].T @ V[:, ii]
            if np.sign(np.real(dotprod)) < 0:
                V[:, ii] = -V[:, ii]

    return V, D