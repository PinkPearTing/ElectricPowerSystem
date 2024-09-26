import os
import sys
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from Info import Info
from Wires import Wires, TubeWire
from Ground import Ground
from Device import Devices
import numpy as np
from scipy.linalg import block_diag
from Utils.Matrix import expand_matrix, copy_and_expand_matrix, update_matrix, update_and_sum_matrix
from Function.Calculators.Impedance import calculate_OHL_wire_impedance
from Vector_Fitting.Calculators.vecfit_kernel_z import vecfit_kernel_Z_Ding
from Vector_Fitting.Drivers.VFdriver import VFdriver
from Vector_Fitting.Drivers.RPdriver import RPdriver
import warnings


class Tower:
    def __init__(self, name, Info, Wires: Wires, tubeWire: TubeWire, Lump, Ground: Ground, Devices: Devices,
                 Measurement):
        """
        初始化杆塔对象

        参数:
        info (TowerInfo): 杆塔自描述信息对象
        wires (Wires): 杆塔切分后线段对象集合
        tubeWire (TubeWire): 管状线段(杆塔中的管状线段唯一, 此处留存初始未切分的管状线段, 方便后续使用, 切分后的多个管状线存储于wires中)
        lump (Circuit): 集中参数对象集合
        ground (Ground): 杆塔地线对象集合
        device (Device): 杆塔设备对象集合
        measurementNode (MeasurementNode): 杆塔测量节点对象集合

        无需传入的参数：
        nodesList (list): 杆塔节点名字列表
        nodesPositions (list): 杆塔节点坐标对列表
        incidence_matrix (numpy.ndarray, Num(wires) * Num(points)): 邻接矩阵
        resistance_matrix (numpy.ndarray, Num(wires) * Num(wires)): 阻抗矩阵
        inductance_matrix (numpy.ndarray, Num(wires) * Num(wires)): 电感矩阵
        potential_matrix (numpy.ndarray, Num(points) * Num(points)): 电位矩阵
        capacitance_matrix (numpy.ndarray, Num(points) * Num(points)): 电容矩阵
        """
        self.name = name
        self.info = Info
        self.wires = Wires
        self.tubeWire = tubeWire
        self.lump = Lump
        self.ground = Ground
        self.devices = Devices
        self.Measurement = Measurement
        self.nodesList = Wires.get_node_names()
        self.nodesPositions = Wires.get_node_coordinates()
        self.bransList = Wires.get_bran_coordinates()
        self.wires_name = []
        # 以下是参数矩阵，是Tower建模最终输出的参数
        # 邻接矩阵
        wires_num = self.wires.count()
        Nodes_num = self.wires.count_distinct_points()
        ag_wires_num = self.wires.count_airWires() + self.wires.count_gndWires()
        self.incidence_matrix = np.zeros((wires_num, Nodes_num))
        # 电阻矩阵
        self.resistance_matrix = np.zeros((ag_wires_num, ag_wires_num))
        # 电感矩阵
        self.inductance_matrix = np.zeros((ag_wires_num, ag_wires_num))
        # 电位矩阵
        self.potential_matrix = np.zeros((Nodes_num, Nodes_num))
        # 电容矩阵
        self.capacitance_matrix = np.zeros((Nodes_num, Nodes_num))
        # 电导矩阵
        self.conductance_matrix = np.zeros((Nodes_num, Nodes_num))
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

    def initialize_incidence_matrix(self):
        """
        initialize_incidence_matrix: calculate the incidence relationship of every wire.
        return: Num(wires) * Num(points) matrix.
                row represents the wire
                column represents the node
                element represents if this node is start point(-1) or end point(+1) for this wire.(neither is 0)
        """
        wire_index = 0
        all_nodes = self.wires.get_all_nodes()
        node_to_index = {node: i for i, node in enumerate(all_nodes)}
        for wire_list in [self.wires.air_wires, self.wires.ground_wires]:
            for wire in wire_list:
                start_node_index = node_to_index[wire.start_node.name]
                end_node_index = node_to_index[wire.end_node.name]
                self.incidence_matrix[wire_index][start_node_index] = -1
                self.incidence_matrix[wire_index][end_node_index] = 1
                wire_index += 1
                self.wires_name.append(wire.name)
        for tube_wire in self.wires.tube_wires:
            start_node_index = node_to_index[tube_wire.sheath.start_node.name]
            end_node_index = node_to_index[tube_wire.sheath.end_node.name]
            self.incidence_matrix[wire_index][start_node_index] = -1
            self.incidence_matrix[wire_index][end_node_index] = 1
            wire_index += 1
            self.wires_name.append(tube_wire.sheath.name)
        for tube_wire in self.wires.tube_wires:
            for core_wire in tube_wire.core_wires:
                start_node_index = node_to_index[core_wire.start_node.name]
                end_node_index = node_to_index[core_wire.end_node.name]
                self.incidence_matrix[wire_index][start_node_index] = -1
                self.incidence_matrix[wire_index][end_node_index] = 1
                wire_index += 1
                self.wires_name.append(core_wire.name)

    def initialize_resistance_matrix(self):
        """
        initialize_resistance_matrix: calculate the resistance of every wire.
        return: n*n diagonal matrix. diagonal elements are resistances of all wires.
        """
        # 1. calculate n*1 matrix for resistance of each wire
        # 2. np.diag(x) will take out the diagonal elements: if x.shape == n*n/n*1
        #    np.diag(x) will use x to create a diagonal matrix: if x.shape == 1*n
        #    so, we should flatten the result of self.wires.get_resistance()* self.wires.get_lengths()
        #    then, np.diag(flattened(x))
        tube_num = self.wires.count_tubeWires()
        resistance = (self.wires.get_resistance() * self.wires.get_lengths()).flatten()
        resistance[-tube_num:] = 0
        self.resistance_matrix = np.diag(resistance)

    def initialize_inductance_matrix(self):
        """
        initialize_inductance_matrix: calculate the inductance of every wire.
        return: n*n diagonal matrix. diagonal elements are resistances of all wires.
        """
        tube_num = self.wires.count_tubeWires()
        inductance = (self.wires.get_inductance() * self.wires.get_lengths()).flatten()
        inductance[-tube_num:] = 0
        self.inductance_matrix = np.diag(inductance)

    def initialize_potential_matrix(self, P):
        Nnags = self.wires.count_distinct_air_gnd_sheathPoints()
        Nnode = self.wires.count_distinct_points()

        self.potential_matrix = np.zeros((Nnode, Nnode))
        self.potential_matrix[:Nnags, :Nnags] = np.copy(P)

    def initialize_capacitance_matrix(self):
        pass

    def initialize_conductance_matrix(self):
        # Nnode = self.wires.count_distinct_air_gnd_sheathPoints()
        # self.conductance_matrix = np.zeros((Nnode, Nnode))
        pass

    def initialize_impedance_matrix(self, L, frequency, constants):
        Nf = frequency.size
        Nbran = self.wires.count_airWires() + self.wires.count_tubeWires() + self.wires.count_gndWires()
        tube_num = self.wires.count_tubeWires()
        length = (self.wires.get_lengths()).flatten()
        length[-tube_num:] = 0

        dz = calculate_OHL_wire_impedance(self.wires.get_radii(), self.wires.get_mur(), self.wires.get_sig(),
                                          self.wires.get_epr(), constants, frequency)
        Z0 = np.zeros((Nbran, Nbran, Nf), dtype=complex)

        for i in range(Nf):
            Zt = np.copy(dz[:, :, i])
            Zt_diag = np.diag(Zt) * length
            np.fill_diagonal(Zt, Zt_diag)
            Z0[:, :, i] = Zt + 1j * 2 * np.pi * frequency[i] * L

        self.impedance_matrix = Z0

    def expand_impedance_matrix(self, Nf):
        # 扩展电阻矩阵
        Nbran = self.wires.count_airWires() + self.wires.count_tubeWires() + self.wires.count_gndWires()
        Nbcore = len(self.wires.tube_wires) * self.wires.tube_wires[0].inner_num

        impedance = np.zeros((Nbran + Nbcore, Nbran + Nbcore, Nf), dtype=complex)

        for i in range(Nf):
            impedance[:Nbran, :Nbran, i] = self.impedance_matrix[:, :, i]

        self.impedance_matrix = impedance

    def update_impedance_matrix_by_tubeWires(self, Zcf, Zsf, Zcsf, Zscf, Lin, sheath_inductance_matrix, length, frequency):
        index = self.wires.get_tubeWires_start_index()
        # 获取索引增量，保证下面循环过程中，index+increment就是下一条管状线段的表皮和芯线的索引
        increment = self.wires.get_tubeWires_index_increment()

        Nf = frequency.size
        Npha = self.wires.tube_wires[0].inner_num

        L0 = np.copy(Lin)
        L0[0, 0] = 0

        for i in range(len(self.wires.tube_wires)):
            for jk in range(Nf):
                Zss = Zsf[0, 0, jk] + 1j * 2 * np.pi * frequency[jk] * sheath_inductance_matrix[i, i]
                Z1 = np.block([[0, Zscf[:, :, jk]], [Zcsf[:, :, jk], Zcf[:, :, jk]]])*length
                Z2 = 1j * 2 * np.pi * frequency[jk] * L0 * length
                Z3 = np.tile(Zscf[:, :, jk], (Npha, 1)) + np.tile(Zcsf[:, :, jk], (1, Npha))
                Z3 = np.block([[0, np.zeros((1, Npha))], [np.zeros((Npha, 1)), Z3]]) * length
                self.impedance_matrix[:, :, jk] = update_matrix(self.impedance_matrix[:, :, jk], index, Z1 + Z2 + Z3 + Zss)
            index = [x + y for x, y in zip(index, increment)]  # index+increment就是下一条管状线段的表皮和芯线的索引

    def add_inductance_matrix(self, L):
        self.inductance_matrix += L

    def update_potential_matrix_by_ground(self, ground_epr):
        Nnas = self.wires.count_distinct_air_sheathPoints()
        Nnags = self.wires.count_distinct_air_gnd_sheathPoints()
        self.potential_matrix[Nnas:Nnags] = self.potential_matrix[Nnas:Nnags] / ground_epr

    def update_potential_matrix_by_tubeWires(self):
        Nnags = self.wires.count_distinct_air_gnd_sheathPoints()
        Nnode = self.wires.count_distinct_points()
        max_potential = self.potential_matrix.max()
        potential_matrix = np.eye(Nnode-Nnags) * max_potential
        self.potential_matrix[Nnags:Nnode, Nnags:Nnode] = potential_matrix

    def update_conductance_matrix_by_ground(self, P, ground_sig, constants):
        ep0 = constants.ep0
        k = ep0 / ground_sig
        Nnode = self.wires.count_distinct_air_gnd_sheathPoints()
        Nnas = self.wires.count_distinct_air_sheathPoints()
        Nnags = self.wires.count_distinct_air_gnd_sheathPoints()

        self.conductance_matrix[Nnas:Nnags, :Nnode] = k * P[Nnas:Nnags]

    def expand_inductance_matrix(self):
        # 通过TubeWire的表皮与其他线段的互感，扩展复制代替为芯线与其他线段的互感，因为芯线和表皮实际上在一个位置
        for i in range(len(self.wires.tube_wires)):
            inner_num = self.wires.tube_wires[i].inner_num
            sheath_index = i + len(self.wires.air_wires) - len(self.wires.tube_wires)
            end_index = len(self.wires.air_wires) + len(self.wires.ground_wires)
            self.inductance_matrix = expand_matrix(self.inductance_matrix, sheath_index, end_index, inner_num)

    def update_inductance_matrix_by_coreWires(self, sheath_inductance_matrix):
        # 获取内部芯线的数量
        inner_num = self.wires.tube_wires[0].inner_num
        # 获取矩阵中表皮开始的索引和结束的索引
        sheath_start_index = len(self.wires.air_wires) + len(self.wires.ground_wires)
        sheath_end_index = len(self.wires.air_wires) + len(self.wires.ground_wires) + len(self.wires.tube_wires)
        # 获取空气和地面支路的结束位置
        end_index = len(self.wires.air_wires) + len(self.wires.ground_wires) + len(self.wires.tube_wires)
        for i in range(sheath_end_index-sheath_start_index):
            temp = np.tile(self.inductance_matrix[sheath_start_index+i, :end_index], (inner_num, 1))
            self.inductance_matrix[end_index+i*inner_num:end_index+(i+1)*inner_num, :end_index] = temp
            temp = np.tile(self.inductance_matrix[:end_index, sheath_start_index + i], (inner_num, 1))
            self.inductance_matrix[:end_index, end_index + i * inner_num:end_index+(i+1)*inner_num] = temp.T

        for i in range(len(self.wires.tube_wires)):
            for j in range(i + 1, len(self.wires.tube_wires)):
                self.inductance_matrix[end_index + i * inner_num:end_index + (i + 1) * inner_num,
                end_index + j * inner_num:end_index + (j + 1) * inner_num] = sheath_inductance_matrix[i, j]
                self.inductance_matrix[end_index + j * inner_num:end_index + (j + 1) * inner_num,
                end_index + i * inner_num:end_index + (i + 1) * inner_num] = sheath_inductance_matrix[j, i]

    def update_inductance_matrix_by_tubeWires(self, sheath_inductance_matrix, Lin, Lx, tube_length):
        # 获取第一个管状线段的表皮和芯线在矩阵中的索引
        index = self.wires.get_tubeWires_start_index()
        # 获取索引增量，保证下面循环过程中，index+increment就是下一条管状线段的表皮和芯线的索引
        increment = self.wires.get_tubeWires_index_increment()

        L0 = Lin.copy()
        L0[0, 0] = 0
        for i in range(len(self.wires.tube_wires)):
            Lss = Lin[0, 0] * tube_length + sheath_inductance_matrix[i, i]
            # L0+Lx+Lss的最终结果 更新到表皮和芯线的自感和互感位置上去
            self.inductance_matrix = update_matrix(self.inductance_matrix, index, L0 * tube_length + Lx * tube_length + Lss)
            index = [x + y for x, y in zip(index, increment)]  # index+increment就是下一条管状线段的表皮和芯线的索引

        return L0

    def expand_resistance_matrix(self):
        # 扩展电阻矩阵
        coreWires_resistance_matrix = np.zeros((len(self.wires.tube_wires) * (self.wires.tube_wires[0].inner_num),
                                                len(self.wires.tube_wires) * (self.wires.tube_wires[0].inner_num)))
        self.resistance_matrix = block_diag(self.resistance_matrix,
                                            coreWires_resistance_matrix)  # 增加芯线的电阻矩阵，此处只做扩充，不做芯线本身的电阻填充

    def update_resistance_matrix_by_tubeWires(self, Rin, Rx, tube_length):
        # 与电感矩阵更新逻辑相同
        index = self.wires.get_tubeWires_start_index()
        increment = self.wires.get_tubeWires_index_increment()

        R0 = Rin.copy()
        R0[0, 0] = 0
        Rss = Rin[0, 0] * tube_length  # 此处与电感矩阵更新过程不同，此处不需要表皮的单位电阻
        for i in range(len(self.wires.tube_wires)):
            self.resistance_matrix = update_matrix(self.resistance_matrix, index, R0*tube_length + Rx * tube_length + Rss)
            index = [x + y for x, y in zip(index, increment)]

    def update_capacitance_matrix_by_tubeWires(self, Cin):
        # 更新电容矩阵
        C0 = update_and_sum_matrix(Cin)
        dist = self.wires.get_tube_lengths()[0]
        indices = self.wires.get_tubeWires_points_index()
        for i in range(len(indices)):
            # 将C矩阵相应位置的点 更新为C0相应位置的数据
            self.capacitance_matrix = update_matrix(self.capacitance_matrix, indices[i], 0.5 * C0 * dist if i == 0 or i == len(
                indices) - 1 else C0 * dist)  # 与外界相连接的部分，需要折半

    def combine_parameter_matrix(self):
        """
        【函数功能】 合并Lumps和Tower的参数矩阵
        """
        # 获取线名称列表
        wire_name_list = self.wires_name

        # 获取节点名称列表
        node_name_list = self.wires.get_all_nodes()

        capacitance_matrix = np.linalg.inv(self.potential_matrix) + self.capacitance_matrix

        df_A = pd.DataFrame(self.incidence_matrix, index=wire_name_list, columns=node_name_list)
        df_R = pd.DataFrame(self.resistance_matrix, index=wire_name_list, columns=wire_name_list)
        df_L = pd.DataFrame(self.inductance_matrix, index=wire_name_list, columns=wire_name_list)
        df_C = pd.DataFrame(capacitance_matrix, index=node_name_list, columns=node_name_list)
        df_G = pd.DataFrame(self.conductance_matrix, index=node_name_list, columns=node_name_list)

        self.incidence_matrix_A = df_A.add(self.lump.incidence_matrix_A, fill_value=0).fillna(0)
        self.incidence_matrix_B = df_A.add(self.lump.incidence_matrix_B, fill_value=0).fillna(0)
        self.resistance_matrix = df_R.add(self.lump.resistance_matrix, fill_value=0).fillna(0)
        self.inductance_matrix = df_L.add(self.lump.inductance_matrix, fill_value=0).fillna(0)
        self.capacitance_matrix = df_C.add(self.lump.capacitance_matrix, fill_value=0).fillna(0)
        self.conductance_matrix = df_G.add(self.lump.conductance_matrix, fill_value=0).fillna(0)
        self.voltage_source_matrix = self.voltage_source_matrix.add(self.lump.voltage_source_matrix,
                                                                    fill_value=0).fillna(0)
        self.current_source_matrix = self.current_source_matrix.add(self.lump.current_source_matrix,
                                                                    fill_value=0).fillna(0)

        del df_A
        del df_R
        del df_L
        del df_G
        del df_C
        if self.devices is not None:
            for device_list in [self.devices.insulators, self.devices.arrestors, self.devices.transformers]:
                for device in device_list:
                    self.incidence_matrix_A = self.incidence_matrix_A.add(device.incidence_matrix_A,
                                                                          fill_value=0).fillna(0)
                    self.incidence_matrix_B = self.incidence_matrix_B.add(device.incidence_matrix_B,
                                                                          fill_value=0).fillna(0)
                    self.resistance_matrix = self.resistance_matrix.add(device.resistance_matrix, fill_value=0).fillna(0)
                    self.inductance_matrix = self.inductance_matrix.add(device.inductance_matrix, fill_value=0).fillna(0)
                    self.capacitance_matrix = self.capacitance_matrix.add(device.capacitance_matrix,
                                                                          fill_value=0).fillna(0)
                    self.conductance_matrix = self.conductance_matrix.add(device.conductance_matrix,
                                                                          fill_value=0).fillna(0)
                    self.voltage_source_matrix = self.voltage_source_matrix.add(device.voltage_source_matrix,
                                                                                fill_value=0).fillna(0)
                    self.current_source_matrix = self.current_source_matrix.add(device.current_source_matrix,
                                                                                fill_value=0).fillna(0)

    def parameter_matrix_update(self):
        """
        【函数功能】 合并Lumps和Tower的参数矩阵
        """
        # 获取线名称列表
        wire_name_list = self.wires_name

        # 获取节点名称列表
        node_name_list = self.wires.get_all_nodes()

        df_A = pd.DataFrame(self.incidence_matrix, index=wire_name_list, columns=node_name_list)
        df_R = pd.DataFrame(self.resistance_matrix, index=wire_name_list, columns=wire_name_list)
        df_L = pd.DataFrame(self.inductance_matrix, index=wire_name_list, columns=wire_name_list)
        df_C = pd.DataFrame(self.capacitance_matrix, index=node_name_list, columns=node_name_list)
        df_G = pd.DataFrame(self.conductance_matrix, index=node_name_list, columns=node_name_list)
        df_V = pd.DataFrame(self.voltage_source_matrix, index=wire_name_list, columns=[0])
        df_I = pd.DataFrame(self.current_source_matrix, index=node_name_list, columns=[0])

        self.incidence_matrix_A = df_A.add(self.lump.incidence_matrix_A, fill_value=0).fillna(0)
        self.incidence_matrix_B = df_A.add(self.lump.incidence_matrix_B, fill_value=0).fillna(0)
        self.resistance_matrix = df_R.add(self.lump.resistance_matrix, fill_value=0).fillna(0)
        self.inductance_matrix = df_L.add(self.lump.inductance_matrix, fill_value=0).fillna(0)
        self.capacitance_matrix = df_C.add(self.lump.capacitance_matrix, fill_value=0).fillna(0)
        self.conductance_matrix = df_G.add(self.lump.conductance_matrix, fill_value=0).fillna(0)
        self.voltage_source_matrix = df_V.add(self.lump.voltage_source_matrix, fill_value=0).fillna(0)
        self.current_source_matrix = df_I.add(self.lump.current_source_matrix, fill_value=0).fillna(0)

        del df_A
        del df_R
        del df_L
        del df_G
        del df_C
        del df_V
        del df_I

        for device_list in [self.devices.insulators, self.devices.arrestors, self.devices.transformers]:
            for device in device_list:
                self.incidence_matrix_A = self.incidence_matrix_A.add(device.incidence_matrix_A,
                                                                      fill_value=0).fillna(0)
                self.incidence_matrix_B = self.incidence_matrix_B.add(device.incidence_matrix_B,
                                                                      fill_value=0).fillna(0)
                self.resistance_matrix = self.resistance_matrix.add(device.resistance_matrix, fill_value=0).fillna(0)
                self.inductance_matrix = self.inductance_matrix.add(device.inductance_matrix, fill_value=0).fillna(0)
                self.capacitance_matrix = self.capacitance_matrix.add(device.capacitance_matrix,
                                                                      fill_value=0).fillna(0)
                self.conductance_matrix = self.conductance_matrix.add(device.conductance_matrix,
                                                                      fill_value=0).fillna(0)
                self.voltage_source_matrix = self.voltage_source_matrix.add(device.voltage_source_matrix,
                                                                            fill_value=0).fillna(0)
                self.current_source_matrix = self.current_source_matrix.add(device.current_source_matrix,
                                                                            fill_value=0).fillna(0)

