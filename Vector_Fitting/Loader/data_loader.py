import scipy.io
import sys, os
sys.path.append(os.pardir)
def load(filename):
    # 加载MATLAB数据文件
    data = scipy.io.loadmat(filename)

    # 访问参数
    Z0 = data['Yorigin']  # 3*3*43的复数矩阵

    frq = data['f0'].flatten()  # 1*43的double类型矩阵，转换为一维数组

    odc = data['Nfit'].flatten()  # 整数

    return data, Z0, frq, odc[0]