import json

import numpy as np
import pandas as pd
from functools import reduce

from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_source, initial_lump, \
    initialize_cable
from Driver.modeling.OHL_modeling import OHL_building
from Driver.modeling.cable_modeling import cable_building
from Driver.modeling.tower_modeling import tower_building

from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire
from Utils.Math import distance
import Model.Strategy as Strategy

class Network:
    def __init__(self, **kwargs):
        self.towers = kwargs.get('towers', [])
        self.cables = kwargs.get('cables', [])
        self.OHLs = kwargs.get('OHLs', [])
        self.sources = kwargs.get('sources', [])
        self.branches = {}
        self.starts = []
        self.ends = []
        self.H = pd.DataFrame()
        self.solution = pd.DataFrame()
        self.incidence_matrix_A = pd.DataFrame()
        self.incidence_matrix_B = pd.DataFrame()
        self.resistance_matrix = pd.DataFrame()
        self.inductance_matrix = pd.DataFrame()
        self.capacitance_matrix = pd.DataFrame()
        self.conductance_matrix = pd.DataFrame()
        self.voltage_source_matrix = pd.DataFrame()
        self.current_source_matrix = pd.DataFrame()
        self.solution_type = {
            'linear': False,
            'constant_step': True,
            'variable_frequency': False
        }
        self.dt = 0
        self.Nt = 0
        self.T = 0

    def calculate_branches(self, maxlength):
        tower_branch_node = {}
        tower_nodes = []
        for tower in self.towers:
            for wire in list(tower.wires.get_all_wires().values()):
                startnode = {wire.start_node.name: [wire.start_node.x, wire.start_node.y, wire.start_node.z]}
                endnode = {wire.end_node.name: [wire.end_node.x, wire.end_node.y, wire.end_node.z]}
                tower_nodes.append(startnode)
                tower_nodes.append(endnode)
                self.branches[wire.name] = [startnode, endnode, tower.name]

        for obj in self.OHLs + self.cables:
            wires = list(obj.wires.get_all_wires().values())
            for wire in wires:
                position_obj_start = {wire.start_node.name: [wire.start_node.x + obj.info.HeadTower_pos[0],
                                                             wire.start_node.y + obj.info.HeadTower_pos[1],
                                                             wire.start_node.z + obj.info.HeadTower_pos[2]]}
                # position_tower_start = self.towers.get(obj.info.HeadTower).info.position
                # start_position = [x + y for x, y in zip(position_obj_start, position_tower_start)]
                position_obj_end = {wire.end_node.name: [wire.end_node.x + obj.info.TailTower_pos[0],
                                                         wire.end_node.y + obj.info.TailTower_pos[1],
                                                         wire.end_node.z + obj.info.TailTower_pos[2]]}
                # position_tower_end = self.towers.get(obj.info.TailTower).info.position
                # end_position = [x + y for x, y in zip(position_obj_end, position_tower_end)]
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                self.branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]

    # initialize internal network elements
    def initialize_network(self,f0,frq_default,max_length,file_name):
        json_file_path = "Data/input/" + file_name + ".json"
        # 0. read json file
        with open(json_file_path, 'r') as j:
            load_dict = json.load(j)

        # 1. initialize all elements in the network
        if 'Tower' in load_dict:
            self.towers = [initialize_tower(tower, max_length=max_length,dt=self.dt,T = self.T) for tower in load_dict['Tower']]
        #self.towers = reduce(lambda a, b: dict(a, **b), tower_list)
        if 'OHL' in load_dict:
            self.OHLs = [initialize_OHL(ohl, max_length=max_length) for ohl in load_dict['OHL']]
        if 'Cable' in load_dict:
            self.cables = [initialize_cable(cable, max_length=max_length) for cable in load_dict['Cable']]

        # 2. build dedicated matrix for all elements
        # segment_num = int(3)  # 正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
        # segment_length = 50  # 预设的参数
        for tower in self.towers:
            tower_building(tower, f0, max_length)

        for ohl in self.OHLs:
            OHL_building(ohl, max_length, frq_default)
        for cable in self.cables:
            cable_building(cable, f0, frq_default)

        # 3. combine matrix
        self.combine_parameter_matrix()

    # initialize external element
    def initialize_source(self, file_name):
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

        for model_list in [self.OHLs, self.cables]:
            for model in model_list:
                self.incidence_matrix_A = self.incidence_matrix_A.add(model.incidence_matrix, fill_value=0).fillna(0)
                self.incidence_matrix_B = self.incidence_matrix_B.add(model.incidence_matrix, fill_value=0).fillna(0)
                self.resistance_matrix = self.resistance_matrix.add(model.resistance_matrix, fill_value=0).fillna(0)
                self.inductance_matrix = self.inductance_matrix.add(model.inductance_matrix, fill_value=0).fillna(0)
                self.capacitance_matrix = self.capacitance_matrix.add(model.capacitance_matrix, fill_value=0).fillna(0)
                self.conductance_matrix = self.conductance_matrix.add(model.conductance_matrix, fill_value=0).fillna(0)
             #   self.voltage_source_matrix.add(model.voltage_source_matrix, fill_value=0).fillna(0)
             #   self.current_source_matrix.add(model.current_source_matrix, fill_value=0).fillna(0)

    def update_H(self, current_result, time):
        for tower in self.towers:
            for ith, lumps in enumerate(
                    [tower.lump] + tower.devices.insulators + tower.devices.arrestors + tower.devices.transformers):
                for component_list in [lumps.switch_disruptive_effect_models, lumps.voltage_controled_switchs]:
                    for component in component_list:
                        v1 = current_result.loc[component.node1, 0].values if component.node1 != 'ref' else 0
                        v2 = current_result.loc[component.node2, 0].values if component.node2 != 'ref' else 0

                        resistance = component.update_parameter(v1, v2)
                        self.resistance_matrix.loc[component.bran[0], component.bran[0]] = resistance

                for time_controled_switch in lumps.time_controled_switchs:
                    resistance = time_controled_switch.update_parameter(time)
                    self.resistance_matrix.loc[
                        time_controled_switch.bran[0], time_controled_switch.bran[0]] = resistance

                for nolinear_resistor in lumps.nolinear_resistors:
                    component_current = abs(current_result.loc[nolinear_resistor.bran, 0].values)
                    resistance = nolinear_resistor.update_parameter(component_current)
                    self.resistance_matrix.loc[nolinear_resistor.bran[0], nolinear_resistor.bran[0]] = resistance

    def calculate(self,strategy):
        strategy.apply(self)

    def run(self,file_name,strategy):
        # 0. 预设值
        frq = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000)
        ])
        VF = {'odc': 10,
              'frq': frq}
        f0 = 2e4 # 固频的频率值
        self.dt = 1e-6
        self.T = 0.001
        self.Nt = int(np.ceil(self.T/self.dt))
        max_length = 20 # 线段的最大长度, 后续会按照这个长度, 对不符合长度规范的线段进行切分

        #1. 初始化电网
        self.initialize_network(f0,frq,max_length,file_name)

        # 2. 初始化源，根据电网信息计算源
        self.calculate_branches(max_length)
        self.initialize_source(file_name)

        # 3. 计算结果
        self.calculate(strategy)
        #network.solution_calculate(dt,Nt)
        #print(x)
        #print(source)

