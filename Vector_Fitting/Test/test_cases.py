import unittest
import sys, os
sys.path.append(os.pardir)
import numpy as np
import scipy.io
from Test.test_utils import assert_dicts_equal
import Drivers.VF_Driver as VFdriver


class TestCases(unittest.TestCase):

    def test_case_1(self):
        # 测试用例1的代码
        data = scipy.io.loadmat("../Data/test.mat")

        # 访问参数
        Z0 = data['Yorigin']  # 3*3*43的复数矩阵

        frq = data['f0'].flatten()  # 1*43的double类型矩阵，转换为一维数组

        odc = data['Nfit'].flatten()  # 整数

        # 准备参数
        VFopts = {'asymp': 3,
                  'plot': 0,
                  'N': odc,
                  'Niter1': 10,
                  'Niter2': 5}
        s = 1j * 2 * np.pi * np.array(frq)
        poles = np.array([])  # Initial poles are automatically generated

        # 调用向量拟合函数，获取输出
        output_data, _, _, _ = VFdriver.drive(Z0, s, poles, VFopts)
        # 获取正确的输出结果
        expected_output = scipy.io.loadmat("../Output/test.mat")

        valid_keys = [k for k in output_data.keys() if k[0] != '_']  # 读取的数据包含不必要的'_header_'等键值对，需要去除

        # 获取两个字段中的有效字段
        filtered_input = {k: output_data[k] for k in valid_keys}
        filtered_expected = {k: expected_output[k] for k in valid_keys}

        # 比对输出是否一致
        assert_dicts_equal(filtered_input, filtered_expected)


    def test_case_2(self):
        data = scipy.io.loadmat("../Data/data.mat")

        # 访问参数
        Z0 = data['Z0']  # 3*3*43的复数矩阵

        frq = data['frq'].flatten()  # 1*43的double类型矩阵，转换为一维数组

        odc = data['odc'].flatten()  # 整数

        # 准备参数
        VFopts = {'asymp': 3,
                  'plot': 0,
                  'N': odc,
                  'Niter1': 10,
                  'Niter2': 5}
        s = 1j * 2 * np.pi * np.array(frq)
        poles = np.array([])  # Initial poles are automatically generated

        # 调用向量拟合函数，获取输出
        output_data, _, _, _ = VFdriver.drive(Z0, s, poles, VFopts)
        # 获取正确的输出结果
        expected_output = scipy.io.loadmat("../Output/data.mat")

        valid_keys = [k for k in output_data.keys() if k[0] != '_']  # 读取的数据包含不必要的'_header_'等键值对，需要去除

        # 获取两个字段中的有效字段
        filtered_input = {k: output_data[k] for k in valid_keys}
        filtered_expected = {k: expected_output[k] for k in valid_keys}

        # 比对输出是否一致
        assert_dicts_equal(filtered_input, filtered_expected)

    # 添加更多测试用例
    ...