import numpy as np
from scipy.special import ber, bei, berp, beip


def calculate_OHL_resistance(resistance, sig, mur, radius, frequency, constants):
    """
    【函数功能】电阻矩阵参数计算
    【入参】
    resistance (numpy.ndarray,n*1): n条线的电阻
    """
    mu0 = constants.mu0
    mu = mur * mu0
    Rd = 1 / (np.pi * sig * radius ** 2)
    if frequency == 0:
        Rc = Rd
    else:
        m = np.sqrt(2 * np.pi * frequency * mu * sig)
        mr = m * radius
        aR = mr / 2 * (ber(mr) * beip(mr) - bei(mr) * berp(mr)) / (berp(mr) ** 2 + beip(mr) ** 2)
        Rc = aR * Rd
    return np.diag(resistance.reshape(-1) + Rc.reshape(-1))