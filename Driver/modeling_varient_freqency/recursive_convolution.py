import numpy as np
import pandas as pd

def preparing_parameters(SER, dt):
    """
    preparing parameters of recursive convolution with vector fitting
    """
    R = SER['R']
    poles = -SER['poles']
    A = np.real(R / poles * (1 - np.exp(-poles * dt)))
    B = np.real(np.exp(-poles * dt))
    return A, B.reshape(1, 1, -1)


def update_phi_matrix(model, I):
    model.phi = model.A.dot(I) + model.B * model.phi