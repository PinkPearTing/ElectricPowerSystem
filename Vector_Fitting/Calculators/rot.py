import numpy as np


def rot(S):
    """
    Rotate the eigenvectors to minimize the error based on imaginary and real parts.

    Parameters:
        S: 2D ndarray, input matrix of eigenvectors

    Returns:
        OUT: 2D ndarray, rotated eigenvectors
    """
    Nc = S.shape[1]
    SA = np.zeros_like(S)
    SB = np.zeros_like(S)

    scale1 = np.zeros(Nc, dtype=complex)
    scale2 = np.zeros(Nc, dtype=complex)
    err1 = np.zeros(Nc)
    err2 = np.zeros(Nc)

    for col in range(Nc):
        numerator = 0.0
        denominator = 0.0

        for j in range(Nc):
            numerator += np.imag(S[j, col]) * np.real(S[j, col])
            denominator += (np.real(S[j, col])) ** 2 - (np.imag(S[j, col])) ** 2

        numerator = -2.0 * numerator
        ang = 0.5 * np.arctan2(numerator, denominator)

        scale1[col] = np.cos(ang) + 1j * np.sin(ang)
        scale2[col] = np.cos(ang + np.pi / 2) + 1j * np.sin(ang + np.pi / 2)

        for j in range(Nc):
            SA[j, col] = S[j, col] * scale1[col]
            SB[j, col] = S[j, col] * scale2[col]

        aaa = bbb = ccc = 0.0
        for j in range(Nc):
            aaa += (np.imag(SA[j, col])) ** 2
            bbb += np.real(SA[j, col]) * np.imag(SA[j, col])
            ccc += (np.real(SA[j, col])) ** 2
        err1[col] = aaa * np.cos(ang) ** 2 + bbb * np.sin(2.0 * ang) + ccc * np.sin(ang) ** 2

        aaa = bbb = ccc = 0.0
        for j in range(Nc):
            aaa += (np.imag(SB[j, col])) ** 2
            bbb += np.real(SB[j, col]) * np.imag(SB[j, col])
            ccc += (np.real(SB[j, col])) ** 2
        err2[col] = aaa * np.cos(ang) ** 2 + bbb * np.sin(2.0 * ang) + ccc * np.sin(ang) ** 2

        if err1[col] < err2[col]:
            scale = scale1[col]
        else:
            scale = scale2[col]

        S[:, col] *= scale

    return S