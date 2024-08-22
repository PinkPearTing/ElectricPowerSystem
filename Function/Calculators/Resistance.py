import numpy as np


def calculate_OHL_resistance(resistance):
    """
    【函数功能】电阻矩阵参数计算
    【入参】
    resistance (numpy.ndarray,n*1): n条线的电阻
    """
    return np.diag(resistance.reshape(-1))