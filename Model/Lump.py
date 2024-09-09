import sys
import numpy as np
import os
import collections
import pandas as pd

from Function.Calculators.Inductance import calculate_OHL_mutual_inductance
from Model.Contant import Constant

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

class Component:
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, parameters: dict = None):
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
            
    def ima_parameter_assign(self, ima):
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        self.assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        self.assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)
            
    def imb_parameter_assign(self, imb):
        """
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        """
        self.assign_incidence_matrix_value(imb, self.bran[0], self.node1[0], -1)
        self.assign_incidence_matrix_value(imb, self.bran[0], self.node2[0], 1)

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r.loc[self.bran[0], self.bran[0]] = self.parameters['resistance']

    def l_parameter_assign(self, l):
        """
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）

        【出参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        l.loc[self.bran[0], self.bran[0]] = self.parameters['inductance']

    def g_parameter_assign(self, g):
        """
        【函数功能】电导参数分配
        【入参】
        g(pandas.Dataframe:Nnode*Nnode)：电导矩阵（Nnode：节点数）
        """
        self.assign_conductance_capcitance_value(g, self.node1[0], self.node2[0], self.parameters['conductance'])

    def c_parameter_assign(self, c):
        """
        【函数功能】电容参数分配
        【入参】
        c(pandas.Dataframe:Nnode*Nnode)：电容矩阵（Nnode：节点数）
        """
        self.assign_conductance_capcitance_value(c, self.node1[0], self.node2[0], self.parameters['capacitor'])


class Resistor_Inductor(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float, inductance: float):
        """
        电阻电感类，继承自 Component 类。

        Args:
            name (str): 电阻电感名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
            inductance (float): 电感值。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "inductance": inductance})


class Conductor_Capacitor(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, conductance: float, capacitor: float):
        """
        电导电容类，继承自 Component 类。

        Args:
            name (str): 电导电容名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            conductance (float): 电导值。
            capacitor (float): 电容值。
        """
        super().__init__(name, bran, node1, node2, {"conductance": conductance, "capacitor": capacitor})


class Voltage_Source_Cosine(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float, magnitude: float,
                 frequency: float, angle: float):
        """
        余弦信号电压源类，继承自 Component 类。

        Args:
            name (str): 电压源名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
            magnitude (float): 幅值。
            frequency (float): 频率。
            angle (float): 相角。
        """
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "magnitude": magnitude, "frequency": frequency, "angle": angle})

    def voltage_calculate(self, calculate_time, dt):
        """
        【函数功能】计算电压源电压
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        voltage(numpy.ndarray,1*(time/dt))：当前电压源电压
        """
        t = np.arange(0, calculate_time, dt)
        voltage = self.parameters['magnitude'] * np.cos(
            2 * np.pi * self.parameters['frequency'] * t + self.parameters['angle'] / 180 * np.pi)
        return voltage


class Voltage_Source_Empirical(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float, voltage: np.ndarray):
        """
        离散信号电压源类，继承自 Component 类。

        Args:
            name (str): 电压源名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
            voltage (numpy.ndarray,1*n): 电压矩阵(n:计算总次数)。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "voltage": voltage})

    def voltage_calculate(self, calculate_time, dt):
        """
        【函数功能】计算电压源电压
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        voltage(numpy.ndarray,1*(time/dt))：当前电压源电压
        """
        calculate_num = int(np.ceil(calculate_time/dt))
        voltage = np.resize(self.parameters['voltage'], calculate_num)
        return voltage


class Current_Source_Cosine(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, magnitude: float,
                 frequency: float, angle: float):
        """
        余弦信号电流源类，继承自 Component 类。

        Args:
            name (str): 电流源名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            magnitude (float): 幅值。
            frequency (float): 频率。
            angle (float): 相角。
        """
        super().__init__(name, bran, node1, node2,
                         {"magnitude": magnitude, "frequency": frequency, "angle": angle})

    def current_calculate(self, calculate_time, dt):
        """
        【函数功能】计算电流源电压
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        current(numpy.ndarray,1*(time/dt))：当前电流源电流
        """
        t = np.arange(0, calculate_time, dt)
        current = self.parameters['magnitude'] * np.cos(
            2 * np.pi * self.parameters['frequency'] * t + self.parameters['angle'] / 180 * np.pi)
        return current


class Current_Source_Empirical(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, current: np.ndarray):
        """
        余弦信号电流源类，继承自 Component 类。

        Args:
            name (str): 电流源名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            current (numpy.ndarray,1*n): 电流矩阵(n:计算总次数)。
        """
        super().__init__(name, bran, node1, node2, {"current": current})

    def current_calculate(self, calculate_time, dt):
        """
        【函数功能】计算电流源电流
        【入参】
        calculate_time(float)：计算总时间，单位：秒
        dt(float)：步长，单位：秒

        【出参】
        current(numpy.ndarray,1*(time/dt))：当前电流源电流
        """
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
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[0], -1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[0], 1)
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[1], self.parameters['gain'])
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[1], -self.parameters['gain'])

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

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r.loc[self.bran[0], self.bran[0]] = self.parameters['resistance']
        r.loc[self.bran[0], self.bran[1]] = self.parameters['gain']


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
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node1[1], -self.parameters['gain'])
        super().assign_incidence_matrix_value(ima, self.bran[0], self.node2[1], self.parameters['gain'])

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r.loc[self.bran[0], self.bran[0]] = 1


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

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r.loc[self.bran[0], self.bran[0]] = 1
        r.loc[self.bran[0], self.bran[1]] = -self.parameters['gain']


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
        r.loc[self.bran[1], self.bran[0]] = self.parameters['ratio']
        r.loc[self.bran[1], self.bran[1]] = 1
        r.loc[self.bran[0], self.bran[1]] *= self.parameters['ratio']


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
        """
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        """
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
            r.loc[self.bran[i + 3], self.bran[i]] = self.parameters['ratio']
            r.loc[self.bran[i + 3], self.bran[i + 3]] = 1
            r.loc[self.bran[i], self.bran[i + 3]] *= self.parameters['ratio']


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
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r.loc[self.bran[0], self.bran[0]] = self.parameters['rmin'] if self.parameters['resistance'][0] == ' ' else \
            self.parameters['resistance'][0]
        r.loc[self.bran[1], self.bran[1]] = self.parameters['rmin'] if self.parameters['resistance'][1] == ' ' else \
            self.parameters['resistance'][1]

    def l_parameter_assign(self, l):
        """
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        l.loc[self.bran[0], self.bran[0]] = self.parameters['inductance'][0, 0]
        l.loc[self.bran[0], self.bran[1]] = self.parameters['inductance'][0, 1]
        l.loc[self.bran[1], self.bran[0]] = self.parameters['inductance'][1, 0]
        l.loc[self.bran[1], self.bran[1]] = self.parameters['inductance'][1, 1]


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
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        for i in range(3):
            r.loc[self.bran[i], self.bran[i]] = self.parameters['rmin'] if self.parameters['resistance'][i] == ' ' else \
                self.parameters['resistance'][i]

    def l_parameter_assign(self, l):
        """
        【函数功能】电感参数分配
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        for i in range(3):
            for j in range(3):
                l.loc[self.bran[i], self.bran[j]] = self.parameters['inductance'][i, j]


class Nolinear_Resistor(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float,
                 vi_characteristic, ri_characteristic, type_of_data: int):
        """
        非线性器件类，继承自 Component 类。

        Args:
            name (str): 非线性电阻名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 默认电阻值。
            vi_characteristic (lambda): 电压-电流特性。
        """
        self.default = 1  # 是否当作非线性电阻的标记
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "vi_characteristic": vi_characteristic,
                          "ri_characteristic": ri_characteristic, "type_of_data": type_of_data})

    def r_parameter_calculate(self, r):
        """
        【函数功能】电阻参数分配（未实现非线性电阻值计算）
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        r.loc[self.bran[0], self.bran[0]] = self.parameters['resistance']

    def update_parameter(self, current):
        """
        【函数功能】参数更新
        """
        resistance = self.parameters['ri_characteristic'](current) if self.default == 1 else self.parameters['resistance']
        resistance = min(resistance, 1e6)
        resistance = max(resistance, 1e-6)
        return resistance


class Nolinear_F(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, inductance: float,
                 bh_characteristic: np.ndarray, type_of_data: int):
        """
        变压器励磁支路类，继承自 Component 类(未完成)。

        Args:
            name (str): 变压器励磁支路名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            inductance (float): 默认电感值。
            bh_characteristic (numpy.ndarray, n*2): b-h特性。
            type_of_data (int): 数据类型。
        """
        self.default = 0  # 是否当作非线性元件的标记
        super().__init__(name, bran, node1, node2,
                         {"inductance": inductance, "bh_characteristic": bh_characteristic,
                          "type_of_data": type_of_data})

    def l_parameter_calculate(self, l):
        """
        【函数功能】电感参数分配（未实现非线性电感值计算）
        【入参】
        l(pandas.Dataframe:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        """
        l.loc[self.bran[0], self.bran[0]] = self.parameters['inductance']


class Voltage_Controled_Switch(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float,
                 voltage: float, type_of_data: int):
        """
        电压控制开关类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
            type_of_data (int): 数据类型。
            voltage(float): 控制电压值
        """
        self.default = 1  # 是否当作非线性元件的标记
        self.on_off = 1 # -1 开关闭合，1 开关断开
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "type_of_data": type_of_data, "voltage": voltage})

    def update_parameter(self, v_primary, v_secondary):
        """
        【函数功能】参数更新计算
        【入参】
        v_primary (float): 首端节点电压。
        v_secondary(float): 末端节点电压
        """
        voltage = abs(v_primary - v_secondary)
        if voltage > self.parameters['voltage']:
            self.on_off = -1
        return self.parameters['resistance'] if self.on_off == 1 else 1e-6


class Time_Controled_Switch(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, close_time: float,
                 open_time: float, type_of_data: int, resistance: float = 1e6):
        """
        电压控制开关类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            type_of_data (int): 数据类型。
            close_time（float）： 闭合时间
            open_time（float）： 断开时间
        """
        self.default = 1  # 是否当作非线性元件的标记
        self.on_off = 3-2 * type_of_data# type_of_data = 1, open2close, on_off = 1; type_of_data = 2, close2open, on_off = -1
        super().__init__(name, bran, node1, node2,
                         {"close_time": close_time, "type_of_data": type_of_data, "open_time": open_time,
                          'resistance': resistance})
    def update_parameter(self, t):
        """
        【函数功能】参数更新计算
        """
        if t > max(self.parameters['close_time'], self.parameters['open_time']):
            self.on_off = 3-2 * self.parameters['type_of_data']
        elif t > min(self.parameters['close_time'], self.parameters['open_time']):
            self.on_off = 2 * self.parameters['type_of_data'] - 3
        return self.parameters['resistance'] if self.on_off == 1 else 1e-6


class A2G(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float):
        """
        电压控制开关类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
        """
        self.default = 0  # 是否当作非线性元件的标记
        super().__init__(name, bran, node1, node2, {"resistance": resistance})


class Ground(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float, inductance: float):
        """
        电阻电感类，继承自 Component 类。

        Args:
            name (str): 电阻电感名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
            inductance (float): 电感值。
        """
        super().__init__(name, bran, node1, node2, {"resistance": resistance, "inductance": inductance})


class Nolinear_Element_Parameters:
    NLE01 = {
        'vi_characteristics': lambda i: (0.09 * np.log10(i) + 0.75) * 42.5 * 10 ** 3,
        'ri_characteristics': lambda i: 42.5 * 10 ** 3 * (0.09 * np.log10(i) + 0.78) / i
    }
    NLE02 = {
        'vi_characteristics': lambda i: (0.08*np.log10(i)+0.61)*42.5*10**3,
        'ri_characteristics': lambda i: 42.5*10**3*(0.08*np.log10(i)+0.61)/i
    }


class Switch_Parameters:
    SWH01 = {
        'DE_max': 140.4e3,
        'v_initial': 168.6e3,
        'k': 1
    }


class Switch_Disruptive_Effect_Model(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, node2: np.ndarray, resistance: float, type_of_data: int,
                 DE_max: float, v_initial: float=168.6e3, k: float=1):
        """
        开关-破坏效应模型类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (np.ndarray, 1*1): 器件支路名称。
            node1 (np.ndarray, 1*1): 器件节点1名称。
            node2 (np.ndarray, 1*1): 器件节点2名称。
            resistance (float): 电阻值。
            type_of_data (int): 数据类型。
            DE_max(float): 破坏效应值。
            v_initial(float): 闪络过程的起始电压。
            k(float): 经验常数。
        """
        self.on_off = 1  # -1 开关闭合，1 开关断开
        self.DE = 0
        super().__init__(name, bran, node1, node2,
                         {"resistance": resistance, "type_of_data": type_of_data, "DE_max": DE_max,
                          "v_initial": v_initial, "k": k})

    def update_parameter(self, voltage, dt):
        """
        【函数功能】破坏效应值计算
        【入参】
        voltage (float): 器件电压。
        """
        if voltage > self.parameters['v_initial']:
            # self.DE += (voltage - self.parameters['v_initial']) ** self.parameters['k']
            self.DE = self.DE + (voltage - self.parameters['v_initial'])*dt

        if self.DE >= self.parameters['DE_max']:
            self.on_off = -1
        return 10 ** (self.on_off * 6)

class MTCK(Component):
    def __init__(self, name: str, bran: np.ndarray, node1: np.ndarray, distance: np.ndarray,
                 high: np.ndarray, radius: np.ndarray):
        """
        传输线阻抗匹配类，继承自 Component 类。

        Args:
            name (str): 电压控制开关名称。
            bran (np.ndarray, n*1): 器件支路名称。
            node1 (np.ndarray, n*1): 器件节点1名称。
            node2 (np.ndarray, n*1): 器件节点2名称。
            distance (float): horizontal distance。
            high (int): 高度。
            DE_max(float): 破坏效应值。
            radius(float): 半径。
        """
        super().__init__(name, bran, node1, None, {"distance": distance, "high": high, "radius": radius})

    def ima_parameter_assign(self, ima):
        """
        【函数功能】关联矩阵A参数分配
        【入参】
        ima(pandas.Dataframe:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        """
        for node in self.node1:
            self.assign_incidence_matrix_value(ima, self.bran[0], node, 1)

    def imb_parameter_assign(self, imb):
        """
        【函数功能】关联矩阵B参数分配
        【入参】
        imb(pandas.Dataframe:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        """
        for node in self.node1:
            self.assign_incidence_matrix_value(imb, self.bran[0], node, 1)

    def r_parameter_assign(self, r):
        """
        【函数功能】电阻参数分配
        【入参】
        r(pandas.Dataframe:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        """
        constants = Constant()
        Vair = constants.Vair
        Lm = calculate_OHL_mutual_inductance(self.parameters['radius'], self.parameters['high'], self.parameters['offset'], constants)
        resistance = Lm*Vair
        r.loc[self.bran, self.bran] = resistance

class Lumps:
    def __init__(self, resistor_inductors=None, conductor_capacitors=None, voltage_control_voltage_sources=None,
                 current_control_voltage_sources=None, voltage_control_current_sources=None,
                 current_control_current_sources=None, transformers_one_phase=None,
                 transformers_three_phase=None, mutual_inductors_two_port=None, mutual_inductors_three_port=None,
                 nolinear_resistors=None, nolinear_fs=None,
                 voltage_controled_switchs=None, time_controled_switchs=None, a2gs=None, grounds=None,
                 switch_disruptive_effect_models=None,
                 current_sources_cosine=None, current_sources_empirical=None, voltage_sources_cosine=None,
                 voltage_sources_empirical=None, MTCKs=None
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
        self.switch_disruptive_effect_models = switch_disruptive_effect_models or []
        self.MTCKs = MTCKs or []
    
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

        for component_list in [self.a2gs, self.resistor_inductors, self.conductor_capacitors,
                               self.current_sources_cosine,
                               self.current_sources_empirical, self.voltage_sources_cosine,
                               self.voltage_sources_empirical, self.switch_disruptive_effect_models,
                               self.nolinear_resistors, self.nolinear_fs, self.voltage_controled_switchs,
                               self.time_controled_switchs, self.grounds]:
            for component in component_list:
                all_nodes[component.node1[0]] = True
                all_nodes[component.node2[0]] = True
                all_brans[component.bran[0]] = True
                
        for component_list in [self.voltage_control_voltage_sources, self.current_control_voltage_sources,
                               self.voltage_control_current_sources, self.current_control_current_sources, 
                               self.transformers_one_phase, self.transformers_three_phase,
                               self.mutual_inductors_two_port, self.mutual_inductors_three_port, self.MTCKs]:
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
                self.current_source_matrix.loc[current_source.node1] = current_source.current_calculate(calculate_time,
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

    def add_MTCK(self, MTCK):
        """
        添加大地
        """
        self.MTCKs.append(MTCK)

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
                self.switch_disruptive_effect_models + self.nolinear_resistors + self.voltage_controled_switchs +
                self.time_controled_switchs + self.MTCKs):
            component.ima_parameter_assign(self.incidence_matrix_A)
            component.imb_parameter_assign(self.incidence_matrix_B)
            component.r_parameter_assign(self.resistance_matrix)

        for component in self.current_control_current_sources:
            component.imb_parameter_assign(self.incidence_matrix_B)
            component.r_parameter_assign(self.resistance_matrix)
