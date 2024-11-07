
import numpy as np
import pandas as pd

from Ground import Ground
from Wires import Wires
from Info import Info

frq_default = np.logspace(0, 9, 37)

class OHL:
    def __init__(self, name: str, Info: Info, Wires: Wires,  Phase, phase_num: int, ground: Ground):
        """
        初始化架空线对象。
        name (str): 线的名称
        info (Info): 自描述信息对象
        Phase (str): 线圈相线
        phase_num (int): 线圈相数
        ground(Ground类)：大地类
        """
        self.name = name
        self.wires = Wires
        self.info = Info
        self.Phase = Phase
        self.phase_num = phase_num
        self.ground = ground
        self.wires_name = []
        self.nodes_name = []
        # 以下是参数矩阵，是OHL建模最终输出的参数
        # 邻接矩阵 (pandas.Dataframe,self.wires_name*self.nodes_name)
        self.incidence_matrix = None
        # 电阻矩阵 (pandas.Dataframe,self.wires_name*self.wires_name)
        self.resistance_matrix = None
        # 电感矩阵 (pandas.Dataframe,self.wires_name*self.wires_name)
        self.inductance_matrix = None
        # 电容矩阵 (pandas.Dataframe,self.nodes_name*self.nodes_name)
        self.capacitance_matrix = None
        # 电导矩阵 (pandas.Dataframe,self.nodes_name*self.nodes_name)
        self.conductance_matrix = None
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

    def get_brans_nodes_list(self, segment_num):
        """
        【函数功能】 获取切分后支路列表与节点列表
        【入参】
        segment_num(int):一条线切分后的线段数
        segment_length(float):线长

        【出参】
        brans_name(list,wires_num*segment_num):支路名称列表
        nodes_name(list,wires_num*segment_num+1):节点名称列表
        """
        brans_name = list(self.wires.get_all_wires().keys())
        start_nodes_name = []
        end_nodes_name = []
        for wire_list in [self.wires.air_wires, self.wires.ground_wires, self.wires.a2g_wires, self.wires.short_wires]:
            for wire in wire_list:
                start_nodes_name.append(wire.start_node.name)
                end_nodes_name.append(wire.end_node.name)

        if segment_num == 1:
            self.wires_name = brans_name
            self.nodes_name = start_nodes_name
            self.nodes_name.extend(end_nodes_name)
        else:
            for i in range(segment_num):
                for j in range(len(brans_name)):
                    self.wires_name.append(f"{brans_name[j]}_Splited_{i+1}")

                self.nodes_name.extend(start_nodes_name)
                for j in range(len(start_nodes_name)):
                    start_nodes_name[j] = f"{brans_name[j]}_MiddleNode_{i + 1}"

            # 最后一个分段，则将终止节点加入列表
            self.nodes_name.extend(end_nodes_name)




