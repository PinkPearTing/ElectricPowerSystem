import json

import numpy as np
import pandas as pd
from functools import reduce

from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_source, initial_lump, \
    initialize_cable, initialize_ground
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
        self.sources = pd.DataFrame()
        self.branches = {}
        self.starts = []
        self.ends = []
        self.H = {}
        self.solution = pd.DataFrame()
        self.measurement = {}
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

        self.f0 = 2e4
        self.max_length = 20
        self.varied_frequency = np.arange(0, 37, 9)
        self.global_ground = 0
        self.ground = None
        self.dt =None
        self.T = None

        self.switch_disruptive_effect_models = []
        self.voltage_controled_switchs = []
        self.time_controled_switchs = []
        self.nolinear_resistors = []
    #记录电网元素之间的关系
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
                position_obj_start = {wire.start_node.name: [wire.start_node.x ,
                                                             wire.start_node.y ,
                                                             wire.start_node.z]}
                # position_tower_start = self.towers.get(obj.info.HeadTower).info.position
                # start_position = [x + y for x, y in zip(position_obj_start, position_tower_start)]
                position_obj_end = {wire.end_node.name: [wire.end_node.x ,
                                                         wire.end_node.y ,
                                                         wire.end_node.z ]}
                # position_tower_end = self.towers.get(obj.info.TailTower).info.position
                # end_position = [x + y for x, y in zip(position_obj_end, position_tower_end)]
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                self.branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]

    # initialize internal network elements
    def initialize_network(self, load_dict, varied_frequency,VF,dt, T):
        self.varied_frequency = varied_frequency


        # 1. initialize all elements in the network
        if 'Tower' in load_dict:
            self.towers = [initialize_tower(tower, max_length=self.max_length,dt=dt,T = T,VF=VF) for tower in load_dict['Tower']]
            self.measurement = reduce(lambda acc,tower:{**acc,**tower.Measurement},self.towers,{})
        #self.towers = reduce(lambda a, b: dict(a, **b), tower_list)
        if 'OHL' in load_dict:
            self.OHLs = [initialize_OHL(ohl, max_length=self.max_length) for ohl in load_dict['OHL']]
        if 'Cable' in load_dict:
            self.cables = [initialize_cable(cable, max_length=self.max_length,VF=VF) for cable in load_dict['Cable']]

        # 2. build dedicated matrix for all elements
        # segment_num = int(3)  # 正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
        # segment_length = 50  # 预设的参数
        for tower in self.towers:
            gnd = self.ground if self.global_ground == 1 else tower.ground
            tower_building(tower, self.f0, self.max_length, gnd, varied_frequency)
            self.switch_disruptive_effect_models.extend(tower.lump.switch_disruptive_effect_models)
            self.voltage_controled_switchs.extend(tower.lump.voltage_controled_switchs)
            self.time_controled_switchs.extend(tower.lump.time_controled_switchs)
            self.nolinear_resistors.extend(tower.lump.nolinear_resistors)
            for device_list in [tower.devices.insulators, tower.devices.arrestors, tower.devices.transformers]:
                for device in device_list:
                    self.switch_disruptive_effect_models.extend(device.switch_disruptive_effect_models)
                    self.voltage_controled_switchs.extend(device.voltage_controled_switchs)
                    self.time_controled_switchs.extend(device.time_controled_switchs)
                    self.nolinear_resistors.extend(device.nolinear_resistors)
        for ohl in self.OHLs:
            gnd = self.ground if self.global_ground == 1 else ohl.ground
            OHL_building(ohl, self.max_length, self.varied_frequency, gnd)
        for cable in self.cables:
            gnd = self.ground if self.global_ground == 1 else cable.ground
            cable_building(cable, self.f0, self.varied_frequency, gnd)

        # 3. combine matrix
        self.combine_parameter_matrix()

    # initialize external source
    def initialize_source(self, load_dict,dt):
        nodes = self.capacitance_matrix.columns.tolist()
        self.sources =initial_source(self, nodes, load_dict, dt=dt)

    def run_measurement(self):
        #lumpname/branname: label,probe,branname,n1,n2,(towername)
        results = {}

        for key, value in self.measurement.items():
            data_type = value[0]  # 0:'branch',1: 'normal lump'
            measurement_type = value[1]  # 1:'current',2:'voltage'

            # 处理支路名或节点名
            if data_type == 0:
                branch_name = key
                if measurement_type == 1:
                    results[key] = self.solution.loc[branch_name].tolist()
                elif measurement_type == 2:
                    node1 = self.solution.get(value[3], None)
                    node2 = self.solution.get(value[4], None)
                    min = [abs(a - b) for a, b in zip(node1, node2)]
                    results[key] = min
            elif data_type == 1:
                lump_name = key
                # 处理支路名可能是列表的情况
                branches = value[2] if isinstance(value[3], list) else [value[3]]
                if measurement_type == 1:
                    dict_result = {}
                    for branch in branches:
                        dict_result[branch] = self.solution.loc[branch].tolist()
                    results[lump_name] = dict_result
                elif measurement_type == 2:
                    dict_result = {}
                    for bran,n1,n2 in zip(value[2],value[3],value[4]):
                        node1 = self.solution.loc[n1].tolist()
                        node2 = self.solution.loc[n2].tolist()
                        min = [abs(a - b) for a, b in zip(node1, node2)]
                        dict_result[bran] = min
                    results[lump_name] = dict_result

        return results




    #R,L,G,C矩阵合并
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
        self.H["incidence_matrix_A"] = self.incidence_matrix_A
        self.H["incidence_matrix_B"] = self.incidence_matrix_B
        self.H["resistance_matrix"] = self.resistance_matrix
        self.H["inductance_matrix"] = self.inductance_matrix
        self.H["capacitance_matrix"] = self.capacitance_matrix
        self.H["conductance_matrix"] = self.conductance_matrix



    #更新H矩阵和判断绝缘子是否闪络
    def update_H(self, current_result, time):
        for switch_v_list in [self.switch_disruptive_effect_models, self.voltage_controled_switchs]:
            for switch_v in switch_v_list:
                v1 = current_result.loc[switch_v.node1[0], 0] if switch_v.node1[0] != 'ref' else 0
                v2 = current_result.loc[switch_v.node2[0], 0] if switch_v.node2[0] != 'ref' else 0
                resistance = switch_v.update_parameter(abs(v1-v2), self.dt)
                self.resistance_matrix.loc[switch_v.bran[0], switch_v.bran[0]] = resistance

        for switch_t in self.time_controled_switchs:
            resistance = switch_t.update_parameter(time)
            self.resistance_matrix.loc[switch_t.bran[0], switch_t.bran[0]] = resistance

        for nolinear_resistor in self.nolinear_resistors:
            component_current = abs(current_result.loc[nolinear_resistor.bran[0], 0])
            resistance = nolinear_resistor.update_parameter(component_current)
            self.resistance_matrix.loc[nolinear_resistor.bran[0], nolinear_resistor.bran[0]] = resistance

    #执行不同的算法
    def calculate(self,strategy,dt):
        strategy.apply(self,dt)

    def run(self,file_name,*basestrategy):

        json_file_path = "Data/input/" + file_name + ".json"
        # 0. read json file
        with open(json_file_path, 'r') as j:
            load_dict = json.load(j)


        # 0. 手动预设值
        frq = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000)
        ])
        VF = {'odc': 10,
              'frq': frq}
        #self.dt = 1e-8
        #self.T = 0.003
        # 是否有定义
        if 'Global' in load_dict:
            self.dt = load_dict['Global']['delta_time']
            self.T = load_dict['Global']['time']
            f0 = load_dict['Global']['constant_frequency']
            max_length = load_dict['Global']['max_length']
            global_ground = load_dict['Global']['ground']['glb']
            ground = initialize_ground(load_dict['Global']['ground']) if 'ground' in load_dict['Global'] else None
        # 2. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict, frq,VF,self.dt,self.T)
        self.Nt = int(np.ceil(self.T/self.dt))
        # 2. 保存支路节点信息
        self.calculate_branches(self.max_length)

        # 3. 初始化源，计算结果
        if load_dict["Source"]["Lightning"]:
            light = load_dict["Source"]["Lightning"]
            self.initialize_source(light,self.dt)

        self.calculate(basestrategy[0],self.dt)

        print("you are measuring")


