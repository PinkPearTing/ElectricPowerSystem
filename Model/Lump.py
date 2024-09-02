import os
import sys
import numpy as np
import os
import time
import shutil
import collections
import pandas as pd
import json

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

class Component:
    def __init__(self, name: str, bran, node1, node2, parameters: dict = None):
        """
        基础Lump元件的抽象基类。

        Args:
            name (str): 元件名称。
            bran (/): 器件支路名称,部分器件存在两条支路。
            node1 (/): 器件节点1名称。
            node2 (/): 器件节点2名称。
            parameters (dict, optional): 元件参数的字典。默认为空字典。
        """
        self.name = name
        self.bran = bran
        self.node1 = node1
        self.node2 = node2
        self.parameters = parameters if parameters is not None else {}

    def assign_incidence_matrix_value(self, im, bran, node, value):
        """
        【函数功能】关联矩阵赋值
        【入参】
        im(pandas.Dataframe:Nbran*Nnode)：关联矩阵（Nbran：支路数，Nnode：节点数）
        bran（str）：支路名称
        node（str）：节点名称
        value(int)：数值

        【出参】
        im(pandas.Dataframe:Nbran*Nnode)：关联矩阵（Nbran：支路数，Nnode：节点数）
        """
        if node != 'ref':
            im.loc[bran, node] = value

    def assign_conductance_capcitance_value(self, gc, node1, node2, value):
        """
        【函数功能】电导电容矩阵赋值
        【入参】
        gc(pandas.Dataframe:Nnode*Nnode)：电导电容矩阵（Nnode：节点数）
        node（str）：节点1名称
        node（str）：节点2名称
        value(int)：数值

        【出参】
        gc(pandas.Dataframe:Nnode*Nnode)：电导电容矩阵（Nnode：节点数）
        """
        if node1 != 'ref':
            gc.loc[node1, node1] += value
        if node2 != 'ref':
            gc.loc[node2, node2] += value
        if node1 != 'ref' and node2 != 'ref':
            gc.loc[node1, node2] -= value
            gc.loc[node2, node1] -= value


class Resistor_Inductor(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float, inductance: float):
        """
        电阻电感类，继承自 Component 类。

        Args:
            name (str): 电阻电感名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
            inductance (float): 电感值。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "inductance": inductance})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(ima, self.bran, self.node2, 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(imb, self.bran, self.node2, 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r.loc[self.bran, self.bran] = self.parameters['resistance']

    def l_parameter_assign(self, l):
        '''
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）

        【出参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        '''
        l.loc[self.bran, self.bran] = self.parameters['inductance']


class Conductor_Capacitor(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, conductance: float, capacitor: float):
        """
        电导电容类，继承自 Component 类。

        Args:
            name (str): 电导电容名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            conductance (float): 电导值。
            capacitor (float): 电容值。
        """
        super().__init__(name, bran, node1, node2, {"conductance": conductance, "capacitor": capacitor})

    def g_parameter_assign(self, g):
        '''
        【函数功能】电导参数分配
        【入参】
        g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
        '''
        super().assign_conductance_capcitance_value(g, self.node1, self.node2, self.parameters['conductance'])

    def c_parameter_assign(self, c):
        '''
        【函数功能】电容参数分配
        【入参】
        c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
        '''
        super().assign_conductance_capcitance_value(c, self.node1, self.node2, self.parameters['capacitor'])


class Voltage_Source_Cosine(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float, magnitude: float,
                 frequency: float, angle: float):
        """
        余弦信号电压源类，继承自 Component 类。

        Args:
            name (str): 电压源名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
            magnitude (float): 幅值。
            frequency (float): 频率。
            angle (float): 相角。
        """
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "magnitude": magnitude, "frequency": frequency, "angle": angle})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(ima, self.bran, self.node2, 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(imb, self.bran, self.node2, 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r.loc[self.bran, self.bran] = self.parameters['resistance']

    def voltage_calculate(self, calculate_time, dt):
        '''
        【函数功能】计算电压源电压
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        voltage(numpy.ndarray,1*(time/dt))：当前电压源电压
        '''
        t = np.arange(0, calculate_time, dt)
        voltage = self.parameters['magnitude'] * np.cos(
            2 * np.pi * self.parameters['frequency'] * t + self.parameters['angle'] / 180 * np.pi)
        return voltage

    def voltage_vector_calculate(self):
        '''
        【函数功能】计算电压源电压向量

        【出参】
        voltage(float)：电压源电压向量
        '''
        voltage = self.parameters['magnitude'] * np.exp(1j * self.parameters['angle'] / 180 * np.pi)
        return voltage


class Voltage_Source_Empirical(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float, voltage: np.ndarray):
        """
        离散信号电压源类，继承自 Component 类。

        Args:
            name (str): 电压源名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
            voltage (numpy.ndarray,1*n): 电压矩阵(n:计算总次数)。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "voltage": voltage})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(ima, self.bran, self.node2, 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(imb, self.bran, self.node2, 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r.loc[self.bran, self.bran] = self.parameters['resistance']

    def voltage_calculate(self, calculate_time, dt):
        '''
        【函数功能】计算电压源电压
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        voltage(numpy.ndarray,1*(time/dt))：当前电压源电压
        '''
        calculate_num = int(np.ceil(calculate_time/dt))
        voltage = np.resize(self.parameters['voltage'], calculate_num)
        return voltage


class Current_Source_Cosine(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, magnitude: float,
                 frequency: float, angle: float):
        """
        余弦信号电流源类，继承自 Component 类。

        Args:
            name (str): 电流源名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            magnitude (float): 幅值。
            frequency (float): 频率。
            angle (float): 相角。
        """
        super().__init__(name, bran, node1, node2,
                         {"magnitude": magnitude, "frequency": frequency, "angle": angle})

    def current_calculate(self, calculate_time, dt):
        '''
        【函数功能】计算电流源电压
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        current(numpy.ndarray,1*(time/dt))：当前电流源电流
        '''
        t = np.arange(0, calculate_time, dt)
        current = self.parameters['magnitude'] * np.cos(
            2 * np.pi * self.parameters['frequency'] * t + self.parameters['angle'] / 180 * np.pi)
        return current

    def current_vector_calculate(self):
        '''
        【函数功能】计算电流源电流向量

        【出参】
        voltage(float)：电流源电流向量
        '''
        current = self.parameters['magnitude'] * np.exp(1j * self.parameters['angle'] / 180 * np.pi)
        return current


class Current_Source_Empirical(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, current: np.ndarray):
        """
        余弦信号电流源类，继承自 Component 类。

        Args:
            name (str): 电流源名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            current (numpy.ndarray,1*n): 电流矩阵(n:计算总次数)。
        """
        super().__init__(name, bran, node1, node2, {"current": current})

    def current_calculate(self, calculate_time, dt):
        '''
        【函数功能】计算电流源电流
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        current(numpy.ndarray,1*(time/dt))：当前电流源电流
        '''
        calculate_num = int(np.ceil(calculate_time/dt))
        current = np.resize(self.parameters['voltage'], calculate_num)
        return current


class Voltage_Control_Voltage_Source(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float,
                 gain: float):
        """
        电压控制电压源类，继承自 Component 类。

        Args:
            name (str): 电压控制电压源名称。
            bran (numpy.ndarray, 1*2): 器件支路名称。
            node1 (numpy.ndarray, 1*2): 器件节点1名称。
            node2 (numpy.ndarray, 1*2): 器件节点2名称。
            resistance (float): 电阻值。
            gain (float): =受控源电压/控制电压。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "gain": gain})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[1], self.parameters['gain'])
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[1], -self.parameters['gain'])

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['resistance']


class Current_Control_Voltage_Source(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float,
                 gain: float):
        """
        电压控制电压源类，继承自 Component 类。

        Args:
            name (str): 电压控制电压源名称。
            bran (numpy.ndarray, 1*2): 器件支路名称。
            node1 (numpy.ndarray, 1*2): 器件节点1名称。
            node2 (numpy.ndarray, 1*2): 器件节点2名称。
            resistance (float): 电阻值。
            gain (float): =受控源电压/控制电流。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "gain": gain})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['resistance']
        r[self.bran[0] - 1, self.bran[1] - 1] = self.parameters['gain']


class Voltage_Control_Current_Source(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, gain: float):
        """
        电压控制电流源类，继承自 Component 类。

        Args:
            name (str): 电压控制电压源名称。
            bran (numpy.ndarray, 1*2): 器件支路名称。
            node1 (numpy.ndarray, 1*2): 器件节点1名称。
            node2 (numpy.ndarray, 1*2): 器件节点2名称。
            gain (float): =受控源电流/控制电压。
        """
        super().__init__(name, bran, node1, node2, {"gain": gain})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[1], -self.parameters['gain'])
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[1], self.parameters['gain'])

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r[self.bran[0] - 1, self.bran[0] - 1] = 1


class Current_Control_Current_Source(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, gain: float):
        """
        电压控制电压源类，继承自 Component 类。

        Args:
            name (str): 电压控制电压源名称。
            bran (numpy.ndarray, 1*2): 器件支路名称。
            node1 (numpy.ndarray, 1*2): 器件节点1名称。
            node2 (numpy.ndarray, 1*2): 器件节点2名称。
            gain (float): =受控源电流/控制电流。
        """
        super().__init__(name, bran, node1, node2, {"gain": gain})

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r[self.bran[0] - 1, self.bran[0] - 1] = 1
        r[self.bran[0] - 1, self.bran[1] - 1] = -self.parameters['gain']


class Transformer_One_Phase(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, Vpri: float,
                 Vsec: float):
        """
        单相理想变压器类，继承自 Component 类。

        Args:
            name (str): 单相理想变压器名称。
            bran (numpy.ndarray, 1*2): 器件支路名称。
            node1 (numpy.ndarray, 1*2): 器件节点1名称。
            node2 (numpy.ndarray, 1*2): 器件节点2名称。
            Vpri (float): 一次侧电压。
            Vsec (float): 二次侧电压。
        """
        self.ratio = Vpri / Vsec
        super().__init__(name, bran, node1, node2, {"ratio": self.ratio})

    def ima_parameter_assign(self, ima):
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[1], self.parameters['ratio'])
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[1], -self.parameters['ratio'])

    def imb_parameter_assign(self, imb):
        """
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        """
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)
        super().assign_incidence_matrix_value(imb, self.bran[1], self.node1[1], -1)
        super().assign_incidence_matrix_value(imb, self.bran[1], self.node2[1], 1)

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r[self.bran[1] - 1, self.bran[0] - 1] = self.parameters['ratio']
        r[self.bran[1] - 1, self.bran[1] - 1] = 1
        r[self.bran[0] - 1, self.bran[1] - 1] *= self.parameters['ratio']


class Transformer_Three_Phase(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, Vpri: float,
                 Vsec: float):
        """
        三相理想变压器类，继承自 Component 类。

        Args:
            name (str): 三相理想变压器名称。
            bran (numpy.ndarray, 1*6): 器件支路名称。
            node1 (numpy.ndarray, 1*6): 器件节点1名称。
            node2 (numpy.ndarray, 1*6): 器件节点2名称。
            Vpri (float): 一次侧电压。
            Vsec (float): 二次侧电压。
        """
        ratio = Vpri / Vsec
        super().__init__(name, bran, node1, node2, {"ratio": ratio})

    def ima_parameter_assign(self, ima):
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        for i in range(3):
            super().assign_incidence_matrix_value(ima, self.bran[i], self.node1[i], -1)
            super().assign_incidence_matrix_value(ima, self.bran[i], self.node2[i], 1)
            super().assign_incidence_matrix_value(ima, self.bran[i], self.node1[i + 3], self.parameters['ratio'])
            super().assign_incidence_matrix_value(ima, self.bran[i], self.node2[i + 3], -self.parameters['ratio'])

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        for i in range(3):
            super().assign_incidence_matrix_value(imb, self.bran[i], self.node1[i], -1)
            super().assign_incidence_matrix_value(imb, self.bran[i], self.node2[i], 1)
            super().assign_incidence_matrix_value(imb, self.bran[i + 3], self.node1[i + 3], -1)
            super().assign_incidence_matrix_value(imb, self.bran[i + 3], self.node2[i + 3], 1)

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        for i in range(3):
            r[self.bran[i + 3] - 1, self.bran[i] - 1] = self.parameters['ratio']
            r[self.bran[i + 3] - 1, self.bran[i + 3] - 1] = 1
            r[self.bran[i] - 1, self.bran[i + 3] - 1] *= self.parameters['ratio']


class Mutual_Inductance_Two_Port(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance,
                 inductance: np.ndarray, rmin=1e-6):
        """
        两端口互感类，继承自 Component 类。

        Args:
            name (str): 两端口互感名称。
            bran (numpy.ndarray, 1*2): 器件支路名称。
            node1 (numpy.ndarray, 1*2): 器件节点1名称。
            node2 (numpy.ndarray, 1*2): 器件节点2名称。
            resistance (numpy.ndarray, 1*2): 电阻矩阵。
            inductance (numpy.ndarray, 2*2): 电感矩阵。
        """
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "inductance": inductance, "rmin": rmin})

    def ima_parameter_assign(self, ima):
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)
        super().assign_incidence_matrix_value(ima, self.bran[1], self.node1[1], -1)
        super().assign_incidence_matrix_value(ima, self.bran[1], self.node2[1], 1)

    def imb_parameter_assign(self, imb):
        """
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        """
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)
        super().assign_incidence_matrix_value(imb, self.bran[1], self.node1[1], -1)
        super().assign_incidence_matrix_value(imb, self.bran[1], self.node2[1], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['rmin'] if self.parameters['resistance'][
                                                                                     0] == ' ' else \
            self.parameters['resistance'][0]
        r[self.bran[1] - 1, self.bran[1] - 1] = self.parameters['rmin'] if self.parameters['resistance'][
                                                                                     1] == ' ' else \
            self.parameters['resistance'][1]

    def l_parameter_assign(self, l):
        """
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        l[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['inductance'][0, 0]
        l[self.bran[0] - 1, self.bran[1] - 1] = self.parameters['inductance'][0, 1]
        l[self.bran[1] - 1, self.bran[0] - 1] = self.parameters['inductance'][1, 0]
        l[self.bran[1] - 1, self.bran[1] - 1] = self.parameters['inductance'][1, 1]


class Mutual_Inductance_Three_Port(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray,
                 resistance: np.ndarray, inductance: np.ndarray, rmin=1e-6):
        """
        三端口互感类，继承自 Component 类。

        Args:
            name (str): 三端口互感名称。
            bran (numpy.ndarray, 1*3): 器件支路名称。
            node1 (numpy.ndarray, 1*3): 器件节点1名称。
            node2 (numpy.ndarray, 1*3): 器件节点2名称。
            resistance (numpy.ndarray, 1*3): 电阻矩阵。
            inductance (numpy.ndarray, 3*3): 电感矩阵。
        """
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "inductance": inductance, "rmin": rmin})

    def ima_parameter_assign(self, ima):
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        for i in range(3):
            super().assign_incidence_matrix_value(ima, self.bran[i], self.node1[i], -1)
            super().assign_incidence_matrix_value(ima, self.bran[i], self.node2[i], 1)

    def imb_parameter_assign(self, imb):
        """
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        """
        for i in range(3):
            super().assign_incidence_matrix_value(imb, self.bran[i], self.node1[i], -1)
            super().assign_incidence_matrix_value(imb, self.bran[i], self.node2[i], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        for i in range(3):
            r[self.bran[i] - 1, self.bran[i] - 1] = self.parameters['rmin'] if self.parameters['resistance'][
                                                                                         i] == ' ' else \
                self.parameters['resistance'][i]

    def l_parameter_assign(self, l):
        """
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        for i in range(3):
            for j in range(3):
                l[self.bran[i] - 1, self.bran[j] - 1] = self.parameters['inductance'][i, j]


class Nolinear_Resistor(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float,
                 vi_characteristic: np.ndarray, type_of_data: int):
        """
        非线性电阻类，继承自 Component 类。

        Args:
            name (str): 非线性电阻名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 默认电阻值。
            vi_characteristic (numpy.ndarray, n*2): 电压-电流特性。
        """
        self.default = 0  # 是否当作非线性电阻的标记
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "vi_characteristic": vi_characteristic,
                          "type_of_data": type_of_data})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_calculate(self, r):
        """
        【函数功能】电阻参数分配（未实现非线性电阻值计算）
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        if self.default == 1:
            r[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['resistance']
        # elif self.default == 0:


class Nolinear_F(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, inductance: float,
                 bh_characteristic: np.ndarray, type_of_data: int):
        """
        变压器励磁支路类，继承自 Component 类。

        Args:
            name (str): 变压器励磁支路名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            inductance (float): 默认电感值。
            bh_characteristic (numpy.ndarray, n*2): b-h特性。
            type_of_data (int): 数据类型。
        """
        self.default = 0  # 是否当作非线性元件的标记
        super().__init__(name, bran, node1, node2,
                         {"inductance": inductance, "bh_characteristic": bh_characteristic,
                          "type_of_data": type_of_data})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def l_parameter_calculate(self, l):
        """
        【函数功能】电感参数分配（未实现非线性电感值计算）
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        if self.default == 1:
            l[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['inductance']
        # elif self.default == 0:


class Voltage_Controled_Switch(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float,
                 voltage: float, type_of_data: int):
        """
        电压控制开关类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
            type_of_data (int): 数据类型。
            voltage(float): 控制电压值
        """
        self.default = 0  # 是否当作非线性元件的标记
        super().__init__(name, bran, node1, node2,
                         {"inductance": resistance, "type_of_data": type_of_data, "voltage": voltage})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)


class Time_Controled_Switch(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, close_time: float,
                 open_time: float, type_of_data: int):
        """
        电压控制开关类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            type_of_data (int): 数据类型。
            close_time（float）： 闭合时间
            open_time（float）： 断开时间
        """
        self.default = 0  # 是否当作非线性元件的标记
        super().__init__(name, bran, node1, node2,
                         {"close_time": close_time, "type_of_data": type_of_data, "open_time": open_time})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)


class A2G(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float):
        """
        电压控制开关类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
        """
        self.default = 0  # 是否当作非线性元件的标记
        super().__init__(name, bran, node1, node2, {"resistance": resistance})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r.loc[self.bran, self.bran] = self.parameters['resistance']


class Ground(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float, inductance: float):
        """
        电阻电感类，继承自 Component 类。

        Args:
            name (str): 电阻电感名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
            inductance (float): 电感值。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "inductance": inductance})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(ima, self.bran, self.node2, 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran, self.node1, -1)
        super().assign_incidence_matrix_value(imb, self.bran, self.node2, 1)

    def r_parameter_assign(self, r):
        '''
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        '''
        r.loc[self.bran, self.bran] = self.parameters['resistance']

    def l_parameter_assign(self, l):
        '''
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        '''
        l.loc[self.bran, self.bran] = self.parameters['inductance']


class Nolinear_Element(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float,
                 vi_characteristic, ri_characteristic, type_of_data: int):
        """
        非线性器件类，继承自 Component 类。

        Args:
            name (str): 非线性电阻名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 默认电阻值。
            vi_characteristic (lambda): 电压-电流特性。
        """
        self.default = 0  # 是否当作非线性电阻的标记
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "vi_characteristic": vi_characteristic,
                          "ri_characteristic": ri_characteristic, "type_of_data": type_of_data})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_calculate(self, r):
        """
        【函数功能】电阻参数分配（未实现非线性电阻值计算）
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        if self.default == 1:
            r[self.bran[0] - 1, self.bran[0] - 1] = self.parameters['resistance']
        # elif self.default == 0:


class Switch_Disruptive_Effect_Model(Component):
    def __init__(self, name: str, bran: str, node1: str, node2: str, resistance: float, type_of_data: int,
                 DE_max: float, v_initial: float=168.6e3, k: float=1):
        """
        开关-破坏效应模型类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (str): 器件支路名称。
            node1 (str): 器件节点1名称。
            node2 (str): 器件节点2名称。
            resistance (float): 电阻值。
            type_of_data (int): 数据类型。
            DE_max(float): 破坏效应值。
            v_initial(float): 闪络过程的起始电压。
            k(float): 经验常数。
        """
        self.on_off = -1  # -1开关闭合，1开关断开
        self.DE = 0
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "type_of_data": type_of_data, "DE_max": DE_max,
                          "v_initial": v_initial, "k": k})

    def ima_parameter_assign(self, ima):
        '''
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)

    def imb_parameter_assign(self, imb):
        '''
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        '''
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def disruptive_effect_calculate(self, v_primary, v_secondary):
        '''
        【函数功能】破坏效应值计算
        【入参】
        v_primary (float): 首端节点电压。
        v_secondary(float): 末端节点电压
        '''
        voltage = abs(v_primary - v_secondary)
        if voltage > self.parameters['v_initial']:
            self.DE += (voltage - self.parameters['v_initial']) ** self.parameters['k']

        if self.DE >= self.parameters['DE_max']:
            self.on_off = 1


class Measurement:
    def __init__(self, name: str, bran_id: int, node1_id: int, node2_id: int, probe: int):
        """
        Measurement的抽象基类。。

        Args:
            name (str): 器件名称。
            bran_id (int): 器件支路名称。
            node1_id (int): 器件节点1名称。
            node2_id (int): 器件节点2名称。
            probe(int):记录参数类型 1：电流，2：电压，3：功率，4：All，5：能量
        """
        self.name = name
        self.bran_id = bran_id
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.probe = probe

    def _case1_initial(self, Nt):
        """
        【函数功能】被测元件电流矩阵初始化
        【入参】
        Nt(int)：计算总次数
        """
        self.I = np.zeros(Nt)

    def _case2_initial(self, Nt):
        """
        【函数功能】被测元件电压矩阵初始化
        【入参】
        Nt(int)：计算总次数
        """
        self.V = np.zeros(Nt)

    def _case3_initial(self, Nt):
        """
        【函数功能】被测元件功率矩阵初始化
        【入参】
        Nt(int)：计算总次数
        """
        self.P = np.zeros(Nt)

    def _case4_initial(self, Nt):
        """
        【函数功能】被测元件电流、电压、功率、能量矩阵初始化
        【入参】
        Nt(int)：计算总次数
        """
        self.I = np.zeros(Nt)
        self.V = np.zeros(Nt)
        self.P = np.zeros(Nt)
        self.E = np.zeros(Nt)

    def _case11_initial(self, Nt):
        """
        【函数功能】被测元件能量矩阵初始化
        【入参】
        Nt(int)：计算总次数
        """
        self.E = np.zeros(Nt)
        self.P_hist = 0

    def initial(self, Nt):
        """
        【函数功能】被测元件参数矩阵初始化

        【入参】
        Nt(int)：计算总次数
        """
        match self.probe:
            case 1:
                self._case1_initial(Nt)
            case 2:
                self._case2_initial(Nt)
            case 3:
                self._case3_initial(Nt)
            case 4:
                self._case4_initial(Nt)
            case 11:
                self._case11_initial(Nt)

    def record_vm(self, v1, v2):
        """
        【函数功能】计算被测元件电压
        【入参】
        v1(float)：器件首端电压
        v2(float)：器件末端电压

        【出参】
        vm(float)：测量电压
        """
        if self.node1_id == 0:
            vm = -v2
        elif self.node2_id == 0:
            vm = v1
        else:
            vm = v1 - v2
        return vm


class Measurement_Linear(Measurement):
    def __init__(self, name: str, bran_id: int, node1_id: int, node2_id: int, probe: int):
        """
        线性元件测量类，继承自 Measurement 类。

        Args:
            name (str): 器件名称。
            bran_id (int): 器件支路名称。
            node1_id (int): 器件节点1名称。
            node2_id (int): 器件节点2名称。
            probe (int): 记录数据种类。
        """
        super().__init__(name, bran_id, node1_id, node2_id, probe)

    def _case1_record(self, out, Nnode, i):
        """
        【函数功能】被测元件电流矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        """
        self.I[i] = out[Nnode + self.bran_id - 1, 0]

    def _case2_record(self, out, i):
        """
        【函数功能】被测元件电压矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        self.V[i] = super().record_vm(v1, v2)

    def _case3_record(self, out, Nnode, i):
        """
        【函数功能】被测元件功率矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = out[Nnode + self.bran_id - 1, 0]
        self.P[i] = vm * im

    def _case4_record(self, out, Nnode, i, dt):
        """
        【函数功能】被测元件电流、电压、功率、能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        dt(float)：步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = out[Nnode + self.bran_id - 1, 0]
        pm = vm * im
        self.I[i] = im
        self.V[i] = vm
        self.P[i] = pm
        if i == 0:
            self.E[i] = 0
        else:
            self.E[i] = self.E[i - 1] + (pm - self.P[i - 1]) * dt

    def _case11_record(self, out, Nnode, i, dt):
        """
        【函数功能】被测元件能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        dt(float)：步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = out[Nnode + self.bran_id - 1, 0]
        pm = vm * im
        if i == 0:
            self.E[i] = 0
        else:
            self.E[i] = self.E[i - 1] + (pm - self.P_hist) * dt
        self.P_hist = pm

    def record_measured_data(self, out, i, dt, Nnode):
        """
        【函数功能】记录被测元件参数
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        dt(float)：步长
        """
        match self.probe:
            case 1:
                self._case1_record(out, Nnode, i)
            case 2:
                self._case2_record(out, i)
            case 3:
                self._case3_record(out, Nnode, i)
            case 4:
                self._case4_record(out, Nnode, i, dt)
            case 11:
                self._case11_record(out, Nnode, i, dt)

    def _case1_steady_record(self, out, Nnode, i):
        """
        【函数功能】稳态计算被测元件电流矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        """
        self.I[i] = np.real(out[Nnode + self.bran_id - 1, 0])

    def _case2_steady_record(self, out, i):
        """
        【函数功能】稳态计算被测元件电压矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        self.V[i] = np.real(super().record_vm(v1, v2))

    def _case3_steady_record(self, out, Nnode, i):
        """
        【函数功能】稳态计算被测元件功率矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = out[Nnode + self.bran_id - 1, 0]
        self.P[i] = np.real(vm * im)

    def _case4_steady_record(self, out, Nnode, i):
        """
        【函数功能】稳态计算被测元件电流、电压、功率、能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = out[Nnode + self.bran_id - 1, 0]
        pm = vm * im
        self.I[i] = np.real(im)
        self.V[i] = np.real(vm)
        self.P[i] = np.real(pm)
        self.E[i] = 0

    def _case11_steady_record(self, out, Nnode, i):
        """
        【函数功能】稳态计算被测元件能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        dt(float)：步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = out[Nnode + self.bran_id - 1, 0]
        pm = vm * im
        self.E[i] = 0
        self.P_hist = np.real(pm)

    def record_steady_measured_data(self, out, i, Nnode):
        """
        【函数功能】稳态计算记录被测元件参数
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        w(float)：计算频率
        """
        match self.probe:
            case 1:
                self._case1_steady_record(out, Nnode, i)
            case 2:
                self._case2_steady_record(out, i)
            case 3:
                self._case3_steady_record(out, Nnode, i)
            case 4:
                self._case4_steady_record(out, Nnode, i)
            case 11:
                self._case11_steady_record(out, Nnode, i)


class Measurement_GC(Measurement):
    def __init__(self, name: str, bran_id: int, node1_id: int, node2_id: int, probe: int,
                 conductance: float, capcitance: float):
        """
        Meas类，继承自 Component 类。

        Args:
            name (str): 器件名称。
            bran_id (int): 器件支路名称。
            node1_id (int): 器件节点1名称。
            node2_id (int): 器件节点2名称。
            probe (int): 记录数据种类。
            conductance (float): 电导值。
            capcitance (float): 电容值。
        """
        self.capcitance = capcitance
        self.conductance = conductance
        self.v_hist = 0
        self.i_hist = 0
        super().__init__(name, bran_id, node1_id, node2_id, probe)

    def _GC_i_calculate(self, vm, dt, diff='backward'):
        """
        【函数功能】计算电容电导的电流
        【入参】
        vm(float)：被测元件电压
        dt（float）：步长
        Diff(str)：差分方法

        【出参】
        Im(float):测量电流
        """
        if diff == 'central' or diff == 'Central':
            Im = self.conductance * vm + 2 * self.capcitance / dt * (vm - self.v_hist) - self.i_hist
        else:
            Im = self.conductance * vm + self.capcitance / dt * (vm - self.v_hist)
        return Im

    def _case1_record(self, out, i, dt, diff='backward'):
        """
        【函数功能】被测元件电流矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        dt(float)：步长
        diff(str):差分方法
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = self._GC_i_calculate(vm, dt, diff)
        self.I[i] = im
        self.v_hist = vm
        self.i_hist = im

    def _case2_record(self, out, i):
        """
        【函数功能】被测元件电压矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        self.V[i] = super().record_vm(v1, v2)

    def _case3_record(self, out, i, dt, diff='backward'):
        """
        【函数功能】被测元件功率矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        dt(float)：步长
        diff(str):差分方法
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = self.record_vm(v1, v2)
        im = self._GC_i_calculate(vm, dt, diff)
        self.P[i] = vm * im
        self.v_hist = vm
        self.i_hist = im

    def _case4_record(self, out, i, dt, diff='backward'):
        """
        【函数功能】被测元件电流、电压、功率、能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        dt(float)：步长
        diff(str):差分方法
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = self.record_vm(v1, v2)
        im = self._GC_i_calculate(vm, dt, diff)
        pm = vm * im
        self.I[i] = im
        self.V[i] = vm
        self.P[i] = pm
        if i == 0:
            self.E[i] = 0
        else:
            self.E[i] = self.E[i - 1] + (pm - self.P[i - 1]) * dt
        self.v_hist = vm
        self.i_hist = im

    def _case11_record(self, out, i, dt, diff='backward'):
        """
        【函数功能】被测元件能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        dt(float)：步长
        diff(str):差分方法
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = self.record_vm(v1, v2)
        im = self._GC_i_calculate(vm, dt, diff)
        pm = vm * im
        if i == 0:
            self.E[i] = 0
        else:
            self.E[i] = self.E[i - 1] + (pm - self.P_hist) * dt
        self.P_hist = pm
        self.v_hist = vm
        self.i_hist = im

    def record_measured_data(self, out, i, dt, diff='backward'):
        """
        【函数功能】记录被测元件参数
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        dt(float)：步长
        Nnode(int)：电路节点数
        diff(str):差分方法
        """
        match self.probe:
            case 1:
                self._case1_record(out, i, dt, diff)
            case 2:
                self._case2_record(out, i)
            case 3:
                self._case3_record(out, i, dt, diff)
            case 4:
                self._case4_record(out, i, dt, diff)
            case 11:
                self._case11_record(out, i, dt, diff),

    def _case1_steady_record(self, out, i, w):
        """
        【函数功能】稳态计算被测元件电流矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        w(float)：计算频率
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = self.conductance * vm + 1j * w * self.capcitance * vm
        self.I[i] = np.real(im)
        self.v_hist = np.copy(np.real(vm))
        self.i_hist = np.copy(np.real(im))

    def _case2_steady_record(self, out, i):
        """
        【函数功能】稳态计算被测元件电压矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        self.V[i] = np.real(super().record_vm(v1, v2))

    def _case3_steady_record(self, out, i, w):
        """
        【函数功能】稳态计算被测元件功率矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = self.conductance * vm + 1j * w * self.capcitance * vm
        self.P[i] = np.real(vm * im)
        self.v_hist = np.real(vm)
        self.i_hist = np.real(im)

    def _case4_steady_record(self, out, i, w):
        """
        【函数功能】稳态计算被测元件电流、电压、功率、能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        dt(float)：步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = self.conductance * vm + 1j * w * self.capcitance * vm
        pm = vm * im
        self.I[i] = np.real(im)
        self.V[i] = np.real(vm)
        self.P[i] = np.real(pm)
        self.v_hist = np.real(vm)
        self.i_hist = np.real(im)

    def _case11_steady_record(self, out, w):
        """
        【函数功能】稳态计算被测元件能量矩阵赋值
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        dt(float)：步长
        """
        v1 = out[self.node1_id - 1, 0]
        v2 = out[self.node2_id - 1, 0]
        vm = super().record_vm(v1, v2)
        im = self.conductance * vm + 1j * w * self.capcitance * vm
        pm = vm * im
        self.P_hist = np.real(pm)
        self.v_hist = np.real(vm)
        self.i_hist = np.real(im)

    def record_steady_measured_data(self, out, i, w):
        """
        【函数功能】稳态计算记录被测元件参数
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        i(int)：当前计算到第i个步长
        w(float)：计算频率
        """
        match self.probe:
            case 1:
                self._case1_steady_record(out, i, w)
            case 2:
                self._case2_steady_record(out, i)
            case 3:
                self._case3_steady_record(out, i, w)
            case 4:
                self._case4_steady_record(out, i, w)
            case 11:
                self._case11_steady_record(out, w)


class Measurements:
    def __init__(self, measurements_linear=None, measurements_gc=None):
        """
        初始化Measurements对象

        参数:
        measurements_linear (Measurement类型的list, optional): 一般线性元件测量列表,默认为空列表。
        measurements_gc (Measurement类型的list, optional): 电导电容元件测量列表,默认为空列表。
        """
        self.measurements_linear = measurements_linear or []
        self.measurements_gc = measurements_gc or []

    def add_measurement_linear(self, measurement_linear):
        """
        添加一般被测元件
        """
        self.measurements_linear.append(measurement_linear)

    def add_measurement_gc(self, measurement_gc):
        """
        添加电导电容被测元件
        """
        self.measurements_gc.append(measurement_gc)

    def record_measured_data(self, out, n, dt, Nnode, diff='backward'):
        """
        【函数功能】记录被测参数

        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        i(int)：当前计算到第i个步长
        dt(float)：步长
        Nnode(int)：电路节点数
        diff(str):差分方法
        """
        for measurement in self.measurements_linear:
            measurement.record_measured_data(out, n, dt, Nnode)

        for measurement in self.measurements_gc:
            measurement.record_measured_data(out, n, dt, diff)

    def record_steady_measured_data(self, out, n, Nnode, w):
        """
        【函数功能】稳态计算记录被测元件参数
        【入参】
        out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
        Nnode(int)：电路节点数
        n(int)：当前计算到第i个步长
        w(float)：计算频率
        """
        for measurement in self.measurements_linear:
            measurement.record_steady_measured_data(out, n, Nnode)

        for measurement in self.measurements_gc:
            measurement.record_steady_measured_data(out, n, w)

    def measurements_initial(self, Nt):
        """
        【函数功能】被测元件初始化
        【入参】
        Nt(int)：计算总长度
        """
        for ith, component in enumerate(self.measurements_linear + self.measurements_gc):
            component.initial(Nt)

class Lumps:
    def __init__(self, resistor_inductors=None, conductor_capacitors=None, voltage_control_voltage_sources=None,
                 current_control_voltage_sources=None, voltage_control_current_sources=None,
                 current_control_current_sources=None, transformers_one_phase=None,
                 transformers_three_phase=None, mutual_inductors_two_port=None, mutual_inductors_three_port=None,
                 sources=None, nolinear_resistors=None, nolinear_fs=None,
                 voltage_controled_switchs=None, time_controled_switchs=None, a2gs=None, grounds=None,
                 measurements=None, nolinear_elements=None, switch_disruptive_effect_models=None,
                 current_sources_cosine=None, current_sources_empirical=None, voltage_sources_cosine=None,
                 voltage_sources_empirical=None
                 ):
        """
        初始化Lumps对象

        参数:
        列表：
        resistor_inductors (Resistor_Inductor类型的list, optional): 电阻电感列表,默认为空列表。
        conductor_capacitors (Conductor_Capacitor类型的list, optional): 电导电容列表,默认为空列表。
        voltage_control_voltage_sources (Voltage_Control_vVltage_Source类型的list, optional): 电压控制电压源列表,默认为空列表。
        current_control_voltage_sources (Current_Control_Voltage_Source类型的list, optional): 电流控制电压源列表,默认为空列表。
        voltage_control_current_sources (Voltage_Control_Current_Source类型的list, optional): 电压控制电流源列表,默认为空列表。
        current_control_current_sources (Current_Control_Current_Source类型的list, optional): 电流控制电流源列表,默认为空列表。
        transformers_one_phase (Transformer_One_Phase类型的list, optional): 单相变压器列表,默认为空列表。
        transformers_three_phase (Transformer_Three_Phase类型的list, optional): 三相变压器列表,默认为空列表。
        mutual_inductors_two_port (Mutual_Inductor_Two_Port类型的list, optional): 两端口互感列表,默认为空列表。
        mutual_inductors_three_port (Mutual_Inductors_Three_Port类型的list, optional): 三端口互感列表,默认为空列表。
        nolinear_resistors (Nolinear_Resistor类型的list, optional): 非线性电阻列表,默认为空列表。
        nolinear_fs (Nolinear_Fs类型的list, optional): 非线性磁通列表,默认为空列表。
        voltage_controled_switchs (Voltage_Controled_Switch类型的list, optional): 电压控制开关列表,默认为空列表。
        time_controled_switchs (Time_Controled_Switch类型的list, optional): 时间控制开关列表,默认为空列表。
        a2gs (A2G类型的list, optional): A2G列表,默认为空列表。
        grounds (Ground类型的list, optional): 大地列表,默认为空列表。
        nolinear_elements (Nolinear_Elements类型的list, optional): 非线性元件列表,默认为空列表。
        switch_disruptive_effect_models(Switch_Disruptive_Effect_Model类型的list, optional): 破坏效应开关列表,默认为空列表。
        voltage_sources_cosine (Voltage_Sources_Cosine类型的list, optional): 余弦电压源列表,默认为空列表。
        voltage_sources_empirical (Voltage_Sources_Empirical类型的list, optional): 离散信号电压源列表,默认为空列表。
        current_sources_cosine (Current_Sources_Cosine类型的list, optional): 余弦电流源列表,默认为空列表。
        current_sources_empirical (Current_Sources_Empirical类型的list, optional): 离散信号电流源列表,默认为空列表。
        类：
        measurements (Measurements类): 被测元件类。
        """
        self.resistor_inductors = resistor_inductors or []
        self.conductor_capacitors = conductor_capacitors or []
        self.voltage_control_voltage_sources = voltage_control_voltage_sources or []
        self.current_control_voltage_sources = current_control_voltage_sources or []
        self.voltage_control_current_sources = voltage_control_current_sources or []
        self.current_control_current_sources = current_control_current_sources or []
        self.transformers_one_phase = transformers_one_phase or []
        self.transformers_three_phase = transformers_three_phase or []
        self.mutual_inductors_two_port = mutual_inductors_two_port or []
        self.mutual_inductors_three_port = mutual_inductors_three_port or []
        self.current_sources_cosine = current_sources_cosine or []
        self.current_sources_empirical = current_sources_empirical or []
        self.voltage_sources_cosine = voltage_sources_cosine or []
        self.voltage_sources_empirical = voltage_sources_empirical or []
        self.nolinear_resistors = nolinear_resistors or []
        self.nolinear_fs = nolinear_fs or []
        self.voltage_controled_switchs = voltage_controled_switchs or []
        self.time_controled_switchs = time_controled_switchs or []
        self.a2gs = a2gs or []
        self.grounds = grounds or []
        self.measurements = measurements or Measurements()
        self.nolinear_elements = nolinear_elements or []
        self.switch_disruptive_effect_models = switch_disruptive_effect_models or []
    
    def brans_nodes_list_initial(self):
        """
        初始化Lump 对象中所有支路和所有节点的集合。

        参数:
        Lumps (Lumps): Lumps 对象

        返回:
        branList(list,Nbran*1)：支路名称列表（Nbran：支路数，Nnode：节点数）
        nodeList(list,Nnode*1)：节点名称列表
        """
        # 获取所有不重复的节点(包含管状线段内部线段的起始点和终止点)
        all_nodes = collections.OrderedDict()
        all_brans = collections.OrderedDict()

        for component_list in [self.resistor_inductors, self.conductor_capacitors, self.current_sources_cosine,
                               self.current_sources_empirical, self.voltage_sources_cosine,
                               self.voltage_sources_empirical]:
            for component in component_list:
                all_nodes[component.node1] = True
                all_nodes[component.node2] = True
                all_brans[component.bran] = True
                
        for component_list in [self.voltage_control_voltage_sources, self.current_control_voltage_sources,
                               self.voltage_control_current_sources, self.current_control_current_sources, 
                               self.transformers_one_phase, self.transformers_three_phase,
                               self.mutual_inductors_two_port, self.mutual_inductors_three_port]:
            for component in component_list:
                for node1 in component.node1:
                    all_nodes[node1] = True
                for node2 in component.node2:
                    all_nodes[node2] = True
                for bran in component.bran:
                    all_brans[bran] = True

        if '---' in all_brans:
            del all_brans['---']
        if 'ref' in all_nodes:
            del all_nodes['ref']
        self.branList = list(all_brans)
        self.nodeList = list(all_nodes)

    def lump_measurement_initial(self, Nt: int):
        """
        【函数功能】Lump测量器件初始化

        【入参】
        Nt(int)：计算总次数
        """
        self.measurements.measurements_initial(Nt)

    def lump_parameter_matrix_initial(self):
        """
        【函数功能】Lump参数矩阵初始化

        【参数】
        branList(list,Nbran*1)：支路名称列表（Nbran：支路数，Nnode：节点数）
        nodeList(list,Nnode*1)：节点名称列表
        incidence_matrix_A(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        incidence_matrix_B(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        resistance_matrix(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        inductance_matrix(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        conductance_matrix(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
        capacitance_matrix(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
        """
        #邻接矩阵A初始化
        self.incidence_matrix_A = pd.DataFrame(0, index=self.branList, columns=self.nodeList, dtype=float)

        # 邻接矩阵B初始化
        self.incidence_matrix_B = pd.DataFrame(0, index=self.branList, columns=self.nodeList, dtype=float)

        # 电阻矩阵R初始化
        self.resistance_matrix = pd.DataFrame(0, index=self.branList, columns=self.branList, dtype=float)

        # 电感矩阵L初始化
        self.inductance_matrix = pd.DataFrame(0, index=self.branList, columns=self.branList, dtype=float)

        # 电导矩阵G初始化
        self.conductance_matrix = pd.DataFrame(0, index=self.nodeList, columns=self.nodeList, dtype=float)

        # 电容矩阵C初始化
        self.capacitance_matrix = pd.DataFrame(0, index=self.nodeList, columns=self.nodeList, dtype=float)

    def lump_voltage_source_matrix_initial(self, calculate_time, dt):
        """
        【函数功能】Lump电压矩阵初始化

        【参数】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒


        【出参】
        voltage_source_matrix(pandas.Dataframe,(Nbran+Nnode)*calculate_num)：电压矩阵
        """
        calculate_num = int(np.ceil(calculate_time / dt))
        # 电源矩阵初始化
        self.voltage_source_matrix = pd.DataFrame(0, index=self.branList, columns=range(calculate_num))

        for voltage_source_list in [self.voltage_sources_cosine, self.voltage_sources_empirical]:
            for voltage_source in voltage_source_list:
                self.voltage_source_matrix.loc[voltage_source.bran] = voltage_source.voltage_calculate(calculate_time,
                                                                                                       dt)

    def lump_current_source_matrix_initial(self, calculate_time, dt):
        """
        【函数功能】Lump电流矩阵初始化

        【参数】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒


        【出参】
        current_source_matrix(pandas.Dataframe,(Nbran+Nnode)*calculate_num)：电liu矩阵
        """
        calculate_num = int(np.ceil(calculate_time / dt))
        # 电源矩阵初始化
        self.current_source_matrix = pd.DataFrame(0, index=self.nodeList, columns=range(calculate_num))

        for current_source_list in [self.current_sources_cosine, self.current_sources_empirical]:
            for current_source in current_source_list:
                self.current_source_matrix.loc[current_source.bran] = current_source.voltage_calculate(calculate_time,
                                                                                               dt)

    def add_current_source_cosine(self, current_source_cosine):
        """
        添加余弦电流源
        """
        self.current_sources_cosine.append(current_source_cosine)

    def add_current_source_empirical(self, current_source_empirical):
        """
        添加离散信号电流源
        """
        self.current_sources_empirical.append(current_source_empirical)

    def add_voltage_source_cosine(self, voltage_source_cosine):
        """
        添加余弦电压源
        """
        self.voltage_sources_cosine.append(voltage_source_cosine)

    def add_voltage_source_empirical(self, voltage_source_empirical):
        """
        添加离散信号电压源
        """
        self.voltage_sources_empirical.append(voltage_source_empirical)

    def add_resistor_inductor(self, resistor_inductor):
        """
        添加电阻电感
        """
        self.resistor_inductors.append(resistor_inductor)

    def add_conductor_capacitor(self, conductor_capacitor):
        """
        添加电导电容
        """
        self.conductor_capacitors.append(conductor_capacitor)

    def add_voltage_control_voltage_source(self, voltage_control_voltage_source):
        """
        添加电压控制电压源
        """
        self.voltage_control_voltage_sources.append(voltage_control_voltage_source)

    def add_current_control_voltage_source(self, current_control_voltage_source):
        """
        添加电流控制电压源
        """
        self.current_control_voltage_sources.append(current_control_voltage_source)

    def add_voltage_control_current_source(self, voltage_control_current_source):
        """
        添加电压控制电流源
        """
        self.voltage_control_current_sources.append(voltage_control_current_source)

    def add_current_control_current_source(self, current_control_current_source):
        """
        添加电流控制电流源
        """
        self.current_control_current_sources.append(current_control_current_source)

    def add_transformer_one_phase(self, transformer_one_phase):
        """
        添加单相变压器
        """
        self.transformers_one_phase.append(transformer_one_phase)

    def add_transformer_three_phase(self, transformer_three_phase):
        """
        添加三相变压器
        """
        self.transformers_three_phase.append(transformer_three_phase)

    def add_mutual_inductor_two_port(self, mutual_inductor_two_port):
        """
        添加两端口互感
        """
        self.mutual_inductors_two_port.append(mutual_inductor_two_port)

    def add_mutual_inductor_three_port(self, mutual_inductor_three_port):
        """
        添加三端口互感
        """
        self.mutual_inductors_three_port.append(mutual_inductor_three_port)

    def add_nolinear_resistor(self, nolinear_resistor):
        """
        添加非线性电阻
        """
        self.nolinear_resistors.append(nolinear_resistor)

    def add_nolinear_element(self, nolinear_element):
        """
        添加非线性元件
        """
        self.nolinear_elements.append(nolinear_element)

    def add_nolinear_f(self, nolinear_f):
        """
        添加非线性磁通
        """
        self.nolinear_fs.append(nolinear_f)

    def add_voltage_controled_switch(self, voltage_controled_switch):
        """
        添加电压控制开关
        """
        self.voltage_controled_switchs.append(voltage_controled_switch)

    def add_time_controled_switch(self, time_controled_switch):
        """
        添加时间控制开关
        """
        self.time_controled_switchs.append(time_controled_switch)

    def add_switch_disruptive_effect_model(self, switch_disruptive_effect_model):
        """
        添加破坏效应模型开关
        """
        self.switch_disruptive_effect_models.append(switch_disruptive_effect_model)

    def add_a2g(self, a2g):
        """
        添加A2G
        """
        self.a2gs.append(a2g)

    def add_ground(self, ground):
        """
        添加大地
        """
        self.grounds.append(ground)

    def parameters_assign(self):
        """
        【函数功能】Lump元件参数分配
        """
        for ith_com, component in enumerate(
                self.resistor_inductors + self.mutual_inductors_two_port + self.mutual_inductors_three_port +
                self.grounds):
            component.ima_parameter_assign(self.incidence_matrix_A)
            component.imb_parameter_assign(self.incidence_matrix_B)
            component.r_parameter_assign(self.resistance_matrix)
            component.l_parameter_assign(self.inductance_matrix)

        for component in self.conductor_capacitors:
            component.g_parameter_assign(self.conductance_matrix)
            component.c_parameter_assign(self.capacitance_matrix)

        for ith_com, component in enumerate(
                self.voltage_control_voltage_sources + self.current_control_voltage_sources +
                self.voltage_control_current_sources + self.transformers_one_phase + self.transformers_three_phase +
                self.a2gs + self.voltage_sources_empirical + self.voltage_sources_cosine +
                self.nolinear_elements):
            component.ima_parameter_assign(self.incidence_matrix_A)
            component.imb_parameter_assign(self.incidence_matrix_B)
            component.r_parameter_assign(self.resistance_matrix)

        for component in self.current_control_current_sources:
            component.imb_parameter_assign(self.incidence_matrix_B)
            component.r_parameter_assign(self.resistance_matrix)


"Lump求解部分，后续需要拆分到其他模块中"
def Lump_zero_initial_solve(ima, imb, R, L, G, C, sources, dt, Meas, diff: str = 'backward'):
    """
    【函数功能】电路零初始状态求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    Meas(Measurements类)：被测元件类
    dt(float)：步长
    diff(str)：差分方法

    【出参】
    out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
    """
    Nn = ima.shape[1]
    SR = sources.source_matrix_update(0, dt)
    LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
    inv_LEFT = np.linalg.inv(LEFT)
    out = inv_LEFT.dot(SR)
    Meas.record_measured_data(out, Nn, dt, 0, diff)

    return out

def Lump_steady_state_solve(ima, imb, R, L, G, C, sources, w, Meas):
    """
    【函数功能】电路稳态求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    Meas(Measurements类)：被测元件类
    w(float):稳态计算频率
    diff(str)：差分方法

    【出参】
    out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
    """
    Nn = ima.shape[1]
    SR = sources.source_vector_matrix_update()
    LEFT = np.block([[-ima, -R - 1j * w * L], [G + 1j * w * C, -imb]])
    inv_LEFT = np.linalg.inv(LEFT)
    out = inv_LEFT.dot(SR)
    Meas.record_steady_measured_data(out, 0, Nn, 2 * np.pi * 50)

    return out

def Lump_central_step_solve(ima, imb, R, L, G, C, sources, n, Ic, dt, hist):
    """
    【函数功能】中心差分法电路求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    n(int)：当前计算到第n个步长
    dt(float)：步长
    hist（numpy.ndarray：Nbran+Nnode*1）:上一时刻计算结果

    【出参】
    out(numpy.ndarray:Nbran+Nnode*1)：中心差分法电路求解结果
    """
    Nb, Nn = ima.shape
    Vs_hist = sources.voltage_matrix_update(n - 1, dt)
    SR = sources.source_matrix_update(n, dt)
    LEFT = np.block([[-ima, -R - 2 * L / dt], [G + 2 * C / dt, -imb]])
    inv_LEFT = np.linalg.inv(LEFT)
    RIGHT = np.block(
        [[(R - 2 * L / dt).dot(hist[Nn:]) + ima.dot(hist[:Nn]) + Vs_hist],
         [(2 * C / dt).dot(hist[:Nn]) + Ic]])
    out = inv_LEFT.dot(SR + RIGHT)
    # Ic = (2 * C / dt).dot(out[:Nn] - hist[:Nn]) - Ic
    return out

def Lump_central_solve(ima, imb, R, L, G, C, sources, Ic, Meas, dt, Nt, initial_state):
    """
    【函数功能】使用中心差分法的电路零初始状态求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    Meas(Measurements类)：被测元件类
    dt(float)：步长
    Nt(int)：计算总次数
    initial_stable（Nbran+Nnode*1）:初始状态计算结果

    【出参】
    结果保存在Measurements类中
    """
    Nnode = ima.shape[1]
    hist = np.copy(initial_state)
    for i in range(Nt - 1):
        out = Lump_central_step_solve(ima, imb, R, L, G, C, sources, i + 1, Ic, dt, hist)
        Ic = (2 * C / dt).dot(out[:Nnode] - hist[:Nnode]) - Ic
        Meas.record_measured_data(out, i + 1, dt, Nnode, 'central')
        hist = np.copy(out)

def Lump_backward_step_solve(ima, imb, R, L, G, C, sources, n, dt, hist):
    """
    【函数功能】使用后向差分法的电路零初始状态求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    n(int)：当前计算到第n个步长
    dt(float)：步长
    hist（numpy.ndarray：Nbran+Nnode*1）:上一时刻计算结果

    【出参】
    out(numpy.ndarray:Nbran+Nnode*1)：电压电流计算结果
    """
    Nn = ima.shape[1]
    SR = sources.source_matrix_update(n, dt)
    LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
    inv_LEFT = np.linalg.inv(LEFT)
    RIGHT = np.block([[(-L / dt).dot(hist[Nn:])], [(C / dt).dot(hist[:Nn])]])
    out = inv_LEFT.dot(SR + RIGHT)
    return out

def Lump_backward_solve(ima, imb, R, L, G, C, sources, Meas, dt, Nt, initial_state):
    """
    【函数功能】电路零初始状态求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    Meas(Measurements类)：被测元件类
    dt(float)：步长
    Nt(int)：计算总次数
    initial_stable（Nbran+Nnode*1）:初始状态计算结果

    【出参】
    结果保存在Measurements类中
    """
    Nnode = ima.shape[1]
    hist = np.copy(initial_state)
    for i in range(Nt - 1):
        out = Lump_backward_step_solve(ima, imb, R, L, G, C, sources, i + 1, dt, hist)
        Meas.record_measured_data(out, Nnode, dt, i + 1, 'backward')
        hist = np.copy(out)


def Lump_circuit_sol(ima, imb, R, L, G, C, sources, measurements, dt, T, Initial_Stable=1, Diff='Central'):
    '''
    【函数功能】电路求解
    【入参】
    ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    sources(Sources类)：电源类
    measurements(Measurements类)：被测元件类
    dt(float)：步长
    T(float)：计算总时间
    Initial_Stable（int）:初始稳态（1：初始稳态，0：零初始状态）
    Diff(str)：差分方法（Central：中心差分，Backward：后向差分）

    【出参】
    结果保存在Measurements类中
    '''
    Nbran, Nnode = ima.shape
    Nt = int(np.ceil(T / dt))
    if Diff == 'central' or Diff == 'Central':
        if Initial_Stable == 0:
            out = Lump_zero_initial_solve(ima, imb, R, L, G, C, sources, dt, measurements)
            Ic = np.real(C.dot(out[:Nnode]))
        else:
            out = Lump_steady_state_solve(ima, imb, R, L, G, C, sources, 2 * np.pi * 50, measurements)
            Ic = np.real(1j * 2 * np.pi * 50 * C.dot(out[:Nnode]))
            out = np.real(out)
        Lump_central_solve(ima, imb, R, L, G, C, sources, Ic, measurements, dt, Nt, out)
    else:
        if Initial_Stable == 0:
            out = Lump_zero_initial_solve(ima, imb, R, L, G, C, sources, dt, measurements)
        else:
            out = Lump_steady_state_solve(ima, imb, R, L, G, C, sources, 2 * np.pi * 50, measurements)
        Lump_backward_solve(ima, imb, R, L, G, C, sources, measurements, dt, Nt, out)


"测试用"


# def GeneInfoRead(data):
#     """
#     读取文件
#     :param data:
#     :return:
#     """
#     DATA = data.copy()
#     DATA.dropna(axis=0, inplace=True, subset=[2, 3, 4])
#     Nrow = DATA.shape[0]
#     row_del = np.array([])
#     for i in range(Nrow):
#         str1 = str(DATA.iloc[i, 0])
#         str2 = str(DATA.iloc[i, 2])
#         if str1 == 'Type':
#             row_del = np.hstack((row_del, DATA.index[i]))
#         elif str1[0] == '%':
#             row_del = np.hstack((row_del, DATA.index[i]))
#         elif str2 == 'nan' or str2 == ' ':
#             row_del = np.hstack((row_del, DATA.index[i]))
#     DATA.drop(row_del, axis=0, inplace=True)
#     DATA.dropna(axis=0, how='all', inplace=True)
#     return DATA.to_numpy()


def NodeBranIndex(data):
    """
    获取节点与支路数
    :param data:
    :return:
    """
    Node = {'list': np.array([]), 'listdex': np.array([], dtype=int), 'num': [0]}
    Bran = {'list': np.array([]), 'listdex': np.array([], dtype=int), 'num': [0]}
    nodebran = np.array([])
    dexn = 0
    dexb = 0
    Nrow = data.shape[0]
    if Nrow != 0:
        namelist = data[:, 2:5]
        Blistdex = []
        Blist = []
        for i in range(Nrow):
            tmp0 = np.zeros(3)
            for j in range(2):
                str = namelist[i, j + 1]
                if str in Node['list']:
                    tmp1 = np.argwhere(Node['list'] == str).squeeze()
                    tmp0[1 + j] = tmp1 + 1
                elif str != 'ref' and str != 'REF':
                    dexn += 1
                    Node['list'] = np.hstack((Node['list'], str))
                    Node['listdex'] = np.hstack((Node['listdex'], dexn))
                    tmp0[1 + j] = dexn
            str = namelist[i, 0]
            if str != '---' and str not in Bran['list']:
                dexb += 1
                Bran['list'] = np.hstack((Bran['list'], namelist[i, 0]))
                Blist.append(namelist[i, :])
                tmp0[0] = dexb
                Blistdex.append(tmp0)
        Bran['list'] = np.array(Blist)
        Bran['listdex'] = np.array(Blistdex, dtype=int)
        Node['num'][0] = dexn
        Bran['num'][0] = dexb
        nodebran = np.copy(Bran['listdex'])
    return Node, Bran, nodebran


def NodeBranIndex_Update(Node, Bran, nodebran_name):
    '''
    【函数功能】获取支路节点ID
    【入参】
    Node(dict:3)：节点编号及ID，节点数量 {list(numpy.ndarray：1*Nnode)；[节点编号]，listdex(numpy.ndarray：Nnode*3)：[节点ID]，num(list:：1)：节点数量}
    Bran(dict:3)：支路编号及ID，支路数量 {list(numpy.ndarray：Nbran*3)：[支路编号，节点编号1，节点编号2]，listdex(numpy.ndarray：Nbran*3)：[支路ID，节点1ID，节点2ID]，num(list：1)：支路数量}
    nodebran_name(pandas.dataframe:N*3)：支路节点编号

    【出参】
    nodebran_name(numpy.ndarray:N*3)：支路节点ID
    '''
    Nrow = nodebran_name.shape[0]
    nodebran_id = np.zeros(nodebran_name.shape, dtype=int)
    for i in range(Nrow):
        str = nodebran_name[i, 0]
        if str in Bran['list'][:, 0]:
            nodebran_id[i, 0] = np.argwhere(Bran['list'][:, 0] == str).squeeze() + 1
        str = nodebran_name[i, 1]
        if str in Node['list']:
            nodebran_id[i, 1] = np.argwhere(Node['list'] == str).squeeze() + 1
        str = nodebran_name[i, 2]
        if str in Node['list']:
            nodebran_id[i, 2] = np.argwhere(Node['list'] == str).squeeze() + 1
    return nodebran_id


def Str2matrix(Str):
    '''
    【函数功能】字符转化矩阵
    【入参】
    Str(pandas.Dataframe:1*1)：参数矩阵 例：1,2;2,3（相同行不同元素用逗号","分割，不同行用分号";"分割）

    【出参】
    matrix(numpy.ndarray:n*n)：参数矩阵
    '''
    str1 = Str.split(';')
    m = len(str1[0].split(','))
    matrix = np.empty((0, m))
    for i in range(len(str1)):
        str2 = np.array(str1[i].split(','), dtype='float')
        matrix = np.vstack((matrix, str2))
    return matrix


class Export_file:
    def __init__(self, root):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
        self.time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.path = os.path.join(root, self.time_now)
        os.makedirs(self.path)

    def create(self, file_name, file, index=False, header=None):
        file = pd.DataFrame(file)
        file_out = os.path.join(self.path, file_name)
        file.to_csv(file_out, index=index, header=header)

    def copy(self, file_list, copyed_file_path):
        for file_name in file_list:
            file = os.path.join(copyed_file_path, file_name)
            if not os.path.isfile(file):
                print('%s not exit!' % (file_name))
            else:
                file_out = os.path.join(self.path, file_name)
                shutil.copy(file, file_out)


def Export_meas(measurements, dir_path):
    """
    将测量结果输出为文件
    保存至指定文件夹的Results文件夹下，以时间.csv命名
    :param measurements:
    :param dir_path:
    :return:
    """
    file_path = os.path.join(dir_path, 'Results')
    ep_file = Export_file(file_path)
    for ith, measurement in enumerate(measurements.measurements_linear + measurements.measurements_gc):
        match measurement.probe:
            case 1:
                file_name = measurement.name + '.csv'
                file = pd.DataFrame(measurement.I.T, columns=['I'])
                ep_file.create(file_name, file, header=True)
            case 2:
                file_name = measurement.name + '.csv'
                file = pd.DataFrame(measurement.V.T, columns=['V'])
                ep_file.create(file_name, file, header=True)
            case 3:
                file_name = measurement.name + '.csv'
                file = pd.DataFrame(measurement.P.T, columns=['P'])
                ep_file.create(file_name, file, header=True)
            case 4:
                file_name = measurement.name + '.csv'
                data = np.vstack(
                    (measurement.I, measurement.V, measurement.P, measurement.E)).T
                file = pd.DataFrame(data, columns=['I', 'V', 'P', 'E'])
                ep_file.create(file_name, file, header=True)
            case 11:
                file_name = measurement.name + '.csv'
                file = pd.DataFrame(measurement.E.T, columns=['E'])
                ep_file.create(file_name, file, header=True)




#
# if __name__ == '__main__':
#     import pandas as pd
#
#     # file_name = '../Data/光伏逆变器.xlsx'
#     # raw_data = pd.read_excel(file_name, index_col=None, header=None)
#     # data = GeneInfoRead(raw_data)
#     # data1 = np.copy(data[0:5, :])
#     # data2 = np.copy(data[5:, :])
#
#     json_file_path = '../Data/01_2.json'
#     # 0. read json file
#     with open(json_file_path, 'r') as j:
#         load_dict = json.load(j)
#     lump_data = load_dict['Lump']
#
#     data1 = lump_data[0:5]
#     data2 = lump_data[5:]
#     lump1 = initial_lump(data1)
#     lump2 = initial_lump(data2)
#
#     a1 = lump1.incidence_matrix_A
#
#     a2 = lump2.incidence_matrix_A
#
#     a_merge = a1.add(a2, fill_value=0).fillna(0)
#
#     print("incidence_matrix_A",a_merge)
#
#     a1 = lump1.incidence_matrix_B
#     a2 = lump2.incidence_matrix_B
#     a_merge = a1.add(a2, fill_value=0).fillna(0)
#
#     print('incidence_matrix_B ?',a_merge)
#
#     a1 = lump1.resistance_matrix
#     a2 = lump2.resistance_matrix
#     a_merge = a1.add(a2, fill_value=0).fillna(0)
#
#     print('resistance_matrix ?',a_merge)
#
#     a1 = lump1.inductance_matrix
#     a2 = lump2.inductance_matrix
#     a_merge = a1.add(a2, fill_value=0).fillna(0)
#
#     print('inductance_matrix ', a_merge)
#
#     a1 = lump1.conductance_matrix
#     a2 = lump2.conductance_matrix
#     a_merge = a1.add(a2, fill_value=0).fillna(0)
#
#     print('conductance_matrix equals?',a_merge)
#
#     a1 = lump1.capacitance_matrix
#     a2 = lump2.capacitance_matrix
#     a_merge = a1.add(a2, fill_value=0).fillna(0)
#
#     print('capacitance_matrix equals?',a_merge)
#
#     # Lump_circuit_sol(lumps.ima, lumps.imb.T, lumps.R, lumps.L, lumps.G, lumps.C, lumps.sources, lumps.measurements, dt,
#     #                  T, Initial_Stable=1, Diff='Central')
#     # Export_meas(lumps.measurements, './')
#     # print(1)
