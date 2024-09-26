import numpy as np
import pandas as pd

from Ground import Ground
from Wires import Wires, TubeWire

class Cable:
    def __init__(self, name,info, Wires: Wires, ground: Ground):
        """
        初始化管状线段对象。(同时满足cable中线段的定义)

        inner_radius (float): 不加套管厚度的内部外径
        outer_radius (float): 添加了套管厚度的整体外径
        inner_num (int): 内部芯线的数量
        """
        self.name = name
        self.info = info
        self.ground = ground
        self.wires = Wires
        self.wires_name = []
        self.nodes_name = []

        self.incidence_matrix = None
        self.resistance_matrix = None
        self.inductancce_matrix = None
        self.capcitance_matrix = None
        self.conductance_matrix = None
        self.Cw = CW(0)
        # 阻抗矩阵
        self.impedance_matrix = np.array([])
        # 电压矩阵
        self.voltage_source_matrix = pd.DataFrame()
        # 电流矩阵
        self.current_source_matrix = pd.DataFrame()
        # vector fitting 相关参数
        self.A = np.array([])
        self.B = np.array([])
        self.phi = np.array([])



class CW:
    def __init__(self, C0):
        self.C0 = C0
