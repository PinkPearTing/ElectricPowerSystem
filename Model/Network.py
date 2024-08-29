import json

import numpy as np
import pandas as pd

from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_source, initial_lump, \
    initialize_cable
from Driver.modeling.OHL_modeling import OHL_building
from Driver.modeling.cable_modeling import cable_building
from Driver.modeling.tower_modeling import tower_building

from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire


class Network:
    def __init__(self, **kwargs):
        self.towers = kwargs.get('towers', [])
        self.cables = kwargs.get('cable', [])
        self.OHLs = kwargs.get('OHLs', [])
        self.sources = kwargs.get('sources', [])
        self.branches = {}
        self.starts = []
        self.ends = []
        self.H = pd.DataFrame()
        self.incidence_matrix_A = pd.DataFrame()
        self.incidence_matrix_B = pd.DataFrame()
        self.resistance_matrix = pd.DataFrame()
        self.inductance_matrix = pd.DataFrame()
        self.capacitance_matrix = pd.DataFrame()
        self.conductance_matrix = pd.DataFrame()
        self.voltage_source_matrix = pd.DataFrame()
        self.current_source_matrix = pd.DataFrame()

    def calculate_branches(self):
        wires = [list(tower.wires.get_all_wires().values()) for tower in self.towers]
        wires2 = [list(ohl.wires_split.get_all_wires().values()) for ohl in self.OHLs]
        wires3 = [list(cable.wires.get_all_wires().values()) for cable in self.cables]
        wires = wires[0] + wires2[0] + wires3[0]

        for wire in wires:
            startnode = [wire.start_node.x, wire.start_node.y, wire.start_node.z]
            endnode = [wire.end_node.x, wire.end_node.y, wire.end_node.z]
            self.branches[wire.name] = [startnode, endnode]
            self.starts.append(startnode)
            self.ends.append(endnode)

    # initialize internal network elements
    def initialize_network(self,f0,frq_default,max_length):
        file_name = "01_2"
        json_file_path = "../Data/" + file_name + ".json"
        # 0. read json file
        with open(json_file_path, 'r') as j:
            load_dict = json.load(j)

        # 1. initialize all elements in the network
        self.towers = [initialize_tower(tower, max_length=max_length) for tower in load_dict['Tower']]
        self.OHLs = [initialize_OHL(ohl, max_length=max_length) for ohl in load_dict['OHL']]
        self.cables = [initialize_cable(cable, max_length=max_length) for cable in load_dict['Cable']]

        # 2. build dedicated matrix for all elements
        # segment_num = int(3)  # 正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
        # segment_length = 50  # 预设的参数
        for tower in self.towers:
            tower_building(tower, f0, max_length)
        for ohl in self.OHLs:
            OHL_building(ohl, frq_default, max_length)
        for cable in self.cables:
            cable_building(cable,f0,frq_default)


        # 3. combine matrix
        self.combine_parameter_matrix()

    # initialize external element
    def initialize_source(self):
        file_name = "01_2"
        nodes = self.capacitance_matrix.columns.tolist()
        self.sources = initial_source(self, nodes, file_name)

    def combine_parameter_matrix(self):
        # 按照towers，cables，ohls顺序合并参数矩阵
        for tower in self.towers:
            self.incidence_matrix_A = self.incidence_matrix_A.add(tower.incidence_matrix_A, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(tower.incidence_matrix_B, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(tower.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(tower.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(tower.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(tower.conductance_matrix, fill_value=0).fillna(0)
           # self.voltage_source_matrix.add(tower.voltage_source_matrix, fill_value=0).fillna(0)
           # self.current_source_matrix.add(tower.current_source_matrix, fill_value=0).fillna(0)

        for model_list in [self.OHLs,self.cables]:
            for model in model_list:
                self.incidence_matrix_A = self.incidence_matrix_A.add(model.incidence_matrix, fill_value=0).fillna(0)
                self.incidence_matrix_B = self.incidence_matrix_B.add(model.incidence_matrix, fill_value=0).fillna(0)
                self.resistance_matrix = self.resistance_matrix.add(model.resistance_matrix, fill_value=0).fillna(0)
                self.inductance_matrix = self.inductance_matrix.add(model.inductance_matrix, fill_value=0).fillna(0)
                self.capacitance_matrix = self.capacitance_matrix.add(model.capacitance_matrix, fill_value=0).fillna(0)
                self.conductance_matrix = self.conductance_matrix.add(model.conductance_matrix, fill_value=0).fillna(0)
             #   self.voltage_source_matrix.add(model.voltage_source_matrix, fill_value=0).fillna(0)
             #   self.current_source_matrix.add(model.current_source_matrix, fill_value=0).fillna(0)


    def concate_H(self):
        GC = self.capacitance_matrix.add(self.conductance_matrix)#点点
        RL = self.inductance_matrix.add(self.resistance_matrix)#线线
        A = self.incidence_matrix_A#线点
        A_T = self.incidence_matrix_B.T#点线

        A_RL = pd.concat([-A, RL], axis=1)
        GC_A = pd.concat([GC, -A_T], axis=1)

        H = pd.concat([A_RL, GC_A], axis=0)

        print("得到一个合并的大矩阵H",H)
        self.H = H

    def H_calculate(self, sources, dt, Nt):
        """
        【函数功能】电路求解
        【入参】
        ima(numpy.ndarray:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        imb(numpy.ndarray:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        R(numpy.ndarray:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        L(numpy.ndarray:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        G(numpy.ndarray:Nnode*Nnode)：电导矩阵（Nnode：节点数）
        C(numpy.ndarray:Nnode*Nnode)：电容矩阵（Nnode：节点数）
        sources(numpy.ndarray:(Nbran+Nnode)*Nt)：电源矩阵（Nbran：支路数，Nnode：节点数）
        dt(float)：步长
        Nt(int)：计算总次数

        【出参】
        out(numpy.ndarray:(Nbran+Nnode)*Nt)：计算结果矩阵（Nbran：支路数，Nnode：节点数）
        """

        G = np.array(self.capacitance_matrix)#点点
        C = np.array(self.conductance_matrix)
        R = np.array(self.inductance_matrix)#线线
        L = np.array(self.resistance_matrix)
        ima = np.array(self.incidence_matrix_A)#线点
        imb = np.array(self.incidence_matrix_B.T)#点线
        nodes = len(self.capacitance_matrix.columns.tolist())
        branches = len(self.inductance_matrix.columns.tolist())
        out = np.zeros((branches + nodes, Nt))
        branches, nodes = ima.shape
        for i in range(Nt - 1):
            Vnode = out[:nodes, i]
            Ibran = out[nodes:, i]
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.pinv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])
            temp_result = inv_LEFT.dot(sources + RIGHT)
            out[:, i + 1] = np.copy(temp_result)
        return out

    def update_H(self):

        print("更新H矩阵",self.H)


if __name__ == '__main__':
    frq = np.concatenate([
        np.arange(1, 91, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, 10000, 1000),
        np.arange(10000, 100000, 10000)
    ])
    VF = {'odc': 10,
          'frq': frq}
    # 固频的频率值
    f0 = 2e4
    dt = 1e-6
    T = 0.001
    Nt = int(np.ceil(T/dt))
    # 线段的最大长度, 后续会按照这个长度, 对不符合长度规范的线段进行切分
    max_length = 50
    network = Network()
    Network.initialize_network(network,f0,frq,max_length)
    network.calculate_branches()
    network.initialize_source()
    # network.concate_H()
    # H_invert = pd.DataFrame(np.linalg.pinv(network.H.values), columns=network.H.index, index=network.H.columns)
    source = network.sources
    # x = H_invert.dot(source)
    out = network.H_calculate(source,dt,Nt)
    #print(x)
    print(out)