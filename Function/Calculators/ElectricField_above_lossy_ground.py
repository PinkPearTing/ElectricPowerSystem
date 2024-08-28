import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from Model.Contant import Constant
from Vector_Fitting.Calculators.vecfit_kernel_z import vecfit_kernel_Z_Ding
def above_lossy(HR0, ER,  constants: Constant, sigma0=None):
    erg = constants.epr
    sigma_g = constants.sigma
    if sigma0 is not None:
        sigma_g = sigma0
    dt = constants.dT
    Nt = constants.Nt

    ep0 = constants.ep0
    u0 = constants.mu0
    vc = constants.vc
    Nd = 9
    w = np.array(
        [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
         10000000])

    H_in = np.zeros(len(w), dtype=complex)
    for ii in range(len(w)):
        H_in[ii] = vc * u0 / np.sqrt(erg + sigma_g / (1j * w[ii] * ep0))

    # The vecfit_kernel_Z_Ding function must be defined or replaced by an equivalent fitting routine
    R0, L0, Rn, Ln, Zfit = vecfit_kernel_Z_Ding(H_in, w / (2 * np.pi), Nd)

    # R0_1 = R0 - np.sum(Rn, axis=2)
    # L0_1 = L0
    # R_1 = Rn[0, 0, :Nd]
    # L_1 = Ln[0, 0, :Nd]

    a00, Nt = HR0.shape
    t_ob = Nt * dt
    conv_2 = 2
    dt0 = dt / conv_2

    x = np.arange(dt, t_ob + dt, dt)
    y = HR0[:, :Nt]
    xi = np.arange(dt0, t_ob + dt0, dt0)
    interp_func = interp1d(x, y, kind='spline', axis=1, fill_value="extrapolate")
    H_save2 = interp_func(xi)

    if a00 == 1:
        H_save2 = H_save2.T

    Ntt = H_save2.shape[1]
    H_all_diff = np.zeros_like(H_save2)
    H_all_diff[:, 0] = H_save2[:, 0] / dt0
    H_all_diff[:, 1:Ntt] = np.diff(H_save2, axis=1) / dt0

    ee0 = R0 * H_save2
    eeL = L0 * H_all_diff

    t00 = Ntt
    Rn2 = Rn[0, :Nd]
    Ln2 = Ln[0, :Nd]
    Rn3 = np.tile(Rn2, (t00, 1))
    Ln3 = np.tile(Ln2, (t00, 1))
    tt00 = np.tile(np.arange(1, t00 + 1).reshape(t00, 1), (1, Nd))
    ee = -Rn3 ** 2 / Ln3 * np.exp(-Rn3 / Ln3 * tt00 * dt0)

    ee_conv = np.zeros((t00, Ntt, Nd))
    for jj in range(Nd):
        ee_conv[:, :, jj] = dt0 * convolve2d(H_save2, ee[:, [jj]].T, mode='full', boundary='fill')[:, :Ntt]

    ee_conv_sum = np.sum(ee_conv, axis=2)
    ee_all = ee0[:, :Ntt:conv_2] + eeL[:, :Ntt:conv_2] + ee_conv_sum[:, :Ntt:conv_2]
    Er_lossy = ER + ee_all.T

    return Er_lossy

