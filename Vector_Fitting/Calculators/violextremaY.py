import numpy as np
from Vector_Fitting.Calculators.fitcalcABCDE import fitcalcABCDE
from Vector_Fitting.Calculators.intercheig import intercheig
from Vector_Fitting.Calculators.rot import rot

def violextremaY(SERflag, wintervals, A, B, C, D, colinterch):
    """
    Identify eigenvalue minima within given intervals.

    Parameters:
        SERflag: Flag for using SER matrices
        wintervals: Array defining intervals for passivity violations
        A, B, C, D: Matrices used in calculations
        colinterch: Flag for column interchange

    Returns:
        s_pass: Frequencies where minima occur
        g_pass: Smallest eigenvalue encountered
        smin: Frequency where g_pass is located
    """
    s_pass = []
    g_pass = []
    smin = []

    if wintervals.size == 0:
        return s_pass, g_pass, smin

    if SERflag == 1:
        SERA = A
        SERC = C
        SERD = D

        Nc = len(SERD)
        N = len(SERA)
        A = np.zeros(Nc * N)

        for col in range(Nc):
            A[col * N: (col + 1) * N] = SERA

        B = np.zeros((Nc * N, Nc))
        for col in range(Nc):
            B[col * N: (col + 1) * N, col] = 1

        C = np.zeros((Nc, Nc * N))
        for row in range(Nc):
            for col in range(Nc):
                C[row, col * N: (col + 1) * N] = SERC[row, col, :N]

        D = SERD
        A = np.diag(A)

    s = []
    Nc = len(D)
    g_pass = 1e16
    smin = 0

    for m in range(len(wintervals)):
        Nint = 21  # Number of internal frequency samples resolving each interval

        w1 = wintervals[m, 0]
        w2 = wintervals[m, 1] if wintervals[m, 1] != 1e16 else 2 * np.pi * 1e16
        s_pass1 = 1j * np.linspace(w1, w2, Nint)

        if w1 == 0:
            s_pass2 = 1j * np.logspace(-8, np.log10(w2), Nint)
        else:
            s_pass2 = 1j * np.logspace(np.log10(w1), np.log10(w2), Nint)

        s_pass = np.sort(np.concatenate([s_pass1, s_pass2]))
        Nint *= 2

        oldT0 = None
        EE = np.zeros((Nc, len(s_pass)), dtype=complex)

        for k in range(len(s_pass)):
            Y = fitcalcABCDE(s_pass[k], np.diag(A), B, C, D, np.zeros(Nc))
            G = np.real(Y)

            if colinterch == 0:
                EE[:, k] = np.linalg.eigvals(G)
            else:
                T0, DD = np.linalg.eig(G)
                T0 = rot(T0)  # Minimizing phase angle of eigenvectors
                T0, DD = intercheig(T0, oldT0, DD, Nc, k)
                oldT0 = T0
                EE[:, k] = np.diag(DD)

        # Identifying violations and picking minima
        s_pass_ind = np.zeros(len(s_pass), dtype=int)
        for row in range(Nc):
            if EE[row, 0] < 0:
                s_pass_ind[0] = 1

        for k in range(1, len(s_pass) - 1):
            for row in range(Nc):
                if EE[row, k] < 0:  # Violation
                    if EE[row, k] < EE[row, k - 1] and EE[row, k] < EE[row, k + 1]:
                        s_pass_ind[k] = 1

        s = np.concatenate([s, s_pass[s_pass_ind == 1]])
        dum = np.min(EE, axis=0)
        g_pass2, ind = np.min(dum), np.argmin(dum)
        smin2 = s_pass[ind]

        g_pass, ind = np.min([g_pass, g_pass2]), np.argmin([g_pass, g_pass2])
        smin = [smin, smin2][ind]
        g_pass = min(g_pass, np.min(EE))

    return s, g_pass, smin