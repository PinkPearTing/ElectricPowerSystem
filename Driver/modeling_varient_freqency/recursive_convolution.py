import numpy as np
import pandas as pd

def preparing_parameters(model, SER, dt):
    """
    preparing parameters of recursive convolution with vector fitting
    """
    R = SER['R']
    poles = SER['poles']
    Nk = SER['R'].shape[-1]
    A = np.real(R / poles * (1 - np.exp(-poles * dt)))
    B = np.real(np.exp(-poles * dt))
    model.A = A
    model.B = B.reshape(1, 1, -1)
    model.phi = np.zeros((len(model.wires_name), 1, Nk))

def update_phi_matrix(model, I):
    model.phi = model.A.dot(I) + model.B * model.phi