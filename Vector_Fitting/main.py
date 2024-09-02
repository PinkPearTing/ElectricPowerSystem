import Loader.data_loader as Loader
import sys, os

sys.path.append(os.pardir)
import numpy as np
import scipy.io as io
import Drivers.VF_Driver as VFdriver


if __name__ == "__main__":
    data = io.loadmat("./Data/data.mat")

    # 访问参数
    Z0 = data['Z0']  # 3*3*43的复数矩阵

    frq = data['frq'].flatten()  # 1*43的double类型矩阵，转换为一维数组

    odc = data['odc'].flatten()  # 整数

    VFopts = {'asymp': 3,
              'plot': 0,
              'N': odc,
              'Niter1': 10,
              'Niter2': 5}
    s = 1j * 2 * np.pi * np.array(frq)

    poles = np.array([])  # Initial poles are automatically generated

    SER, rmserr, bigHfit, opts = VFdriver.drive(Z0, s, poles, VFopts)
    print('finish')
    io.savemat('./Output/data.mat', SER)

