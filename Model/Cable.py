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

    def get_cable_nodes_list(self):
        """
        【函数功能】 获取切分后支路列表与节点列表
        """
        nodes_name = []
        for tubewire in self.wires.tube_wires:
            nodes_name.append(tubewire.sheath.start_node.name)
            for core_wire in tubewire.core_wires:
                nodes_name.append(core_wire.start_node.name)

        nodes_name.append(tubewire.sheath.end_node.name)
        for core_wire in tubewire.core_wires:
            nodes_name.append(core_wire.end_node.name)
        # end_wire_num = len(self.wires_name)
        # nodes_name = self.wires.get_all_nodes()
        # nodes_name.remove('ref')
        # for ith, start_wire in enumerate([self.wires.tube_wires[0].sheath] + self.wires.tube_wires[0].core_wires):
        #     if start_wire.start_node.name == 'ref':
        #         nodes_name.insert(ith, 'ref')
        # for jth, end_wire in enumerate([self.wires.tube_wires[-1].sheath] + self.wires.tube_wires[-1].core_wires):
        #     if end_wire.end_node.name == 'ref':
        #         nodes_name.insert(end_wire_num + jth, 'ref')
        return nodes_name



class CW:
    def __init__(self, C0):
        self.C0 = C0
