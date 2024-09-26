import json
import numpy as np
import copy

from pandas.core.interchange.dataframe_protocol import DataFrame

from Model.Lump import Lumps, Resistor_Inductor, Conductor_Capacitor, \
    Voltage_Source_Cosine, Voltage_Source_Empirical, Current_Source_Cosine, Current_Source_Empirical, \
    Voltage_Control_Voltage_Source, Current_Control_Voltage_Source, Voltage_Control_Current_Source, \
    Current_Control_Current_Source, Transformer_One_Phase, Transformer_Three_Phase, Mutual_Inductance_Two_Port, \
    Mutual_Inductance_Three_Port, Nolinear_Resistor, Nolinear_F, Voltage_Controled_Switch, Time_Controled_Switch, A2G, \
    Switch_Disruptive_Effect_Model, Nolinear_Element_Parameters, Switch_Parameters
from Model.Node import Node
from Model.Wires import Wire, Wires, CoreWire, TubeWire
from Model.Ground import Ground
from Model.Tower import Tower
from Model.OHL import OHL
from Model.Lightning import Stroke, Lightning, Channel
from Model.Contant import Constant
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate, LightningCurrent_calculate
import pandas as pd
from Model.Cable import Cable
from Model.Info import TowerInfo,OHLInfo, CableInfo
from Model.Device import Devices
from Utils.Math import segment_branch

# initialize wire in tower
def initialize_wire(wire, nodes,VF):
    bran = wire['bran']
    node_name_start = wire['node1']
    pos_start = wire['pos_1']
    node_name_end = wire['node2']
    pos_end = wire['pos_2']
    node_start = Node(name=node_name_start, x=pos_start[0], y=pos_start[1], z=pos_start[2])
    node_end = Node(name=node_name_end, x=pos_end[0], y=pos_end[1], z=pos_end[2])

    offset = wire['oft']
    R = wire['r']
    L = wire['l']
    sig = wire['sig']
    mur = wire['mur']
    epr = wire['epr']
    # 自定义一个VF
    # 初始化向量拟合参数

    if wire['type'] == 'air' or wire['type'] == 'sheath' or wire['type'] == 'ground':
        radius = wire['r0']
        return Wire(bran, node_start, node_end, offset, radius, R, L, sig, mur, epr, VF)
    elif wire['type'] == 'core':
        radius = wire['rc']
        return CoreWire(bran, node_start, node_end, offset, radius, R, L, sig, mur, epr, VF, wire['oft'], wire['cita'])

# initialize wire in OHL
def initialize_OHL_wire(wire):
    cir_id = wire['cir_id']
    if wire['type'] == 'SW':
        bran = 'Y' + str(cir_id) + 'S'
    elif wire['type'] == 'CIRO':
        bran = 'Y' + str(cir_id) + wire['phase']
    else:
        bran = wire['bran']
    node_name_start = wire['node1']
    pos_start = wire['node1_pos']
    node_name_end = wire['node2']
    pos_end = wire['node2_pos']
    node_start = Node(name=node_name_start, x=pos_start[0], y=pos_start[1], z=pos_start[2])
    node_end = Node(name=node_name_end, x=pos_end[0], y=pos_end[1], z=pos_end[2])


    offset = wire['offset']
    radius = wire['r0']
    R = wire['r']
    L = wire['l']
    sig = wire['sig']
    mur = wire['mur']
    epr = wire['epr']
    # 自定义一个VF
    # 初始化向量拟合参数
    frq = np.concatenate([
        np.arange(1, 91, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, 10000, 1000),
        np.arange(10000, 100000, 10000),
    ])
    VF = {'odc': 10,
          'frq': frq}

    return Wire(bran, node_start, node_end, offset, radius, R, L, sig, mur, epr, VF)

# initialize ground in tower
def initialize_ground(ground_dic):
    sig = ground_dic['sig']
    mur = ground_dic['mur']
    epr = ground_dic['epr']
    model = ground_dic['gnd_model']
    ionisation_intensity = ground_dic['ionisation_intensity']
    ionisation_model = ground_dic['ionisation_model']

    return Ground(sig, mur, epr, model, ionisation_intensity, ionisation_model)

def initalize_wire_measurement(wire,measurement,tower_name):
    if wire['probe']:
        measurement[wire['bran']] = [0,wire[ 'probe'],wire['bran'],wire['node1'],wire['node2'],tower_name]




def initialize_tower(tower_dict, max_length, dt, T,VF):


    # tower_dict['Wire']
    # 1. initialize wires
    wires = Wires()
    tube_wire = None
    nodes = []
    measurement = {}
    for wire in tower_dict['Wire']:

        initalize_wire_measurement(wire,measurement,tower_dict['Info']['name'])
        # 1.1 initialize air wire
        if wire['type'] == 'air':
            wire_air = initialize_wire(wire, nodes,VF)
            wires.add_air_wire(wire_air)  # add air wire in wires

        # 1.2 initialize ground wire
        elif wire['type'] == 'ground':
            wire_ground = initialize_wire(wire, nodes,VF)
            wires.add_ground_wire(wire_ground)  # add ground wire in wires

        # 1.3 initialize tube
        elif wire['type'] == 'tube':
            sheath_wire = initialize_wire(wire['sheath'], nodes,VF)
            tube_wire = TubeWire(sheath_wire, wire['sheath']['rs1'], wire['sheath']['rs3'], wire['sheath']['num'])

            for core in wire['core']:
                core_wire = initialize_wire(core, nodes,VF)
                tube_wire.add_core_wire(core_wire)

            wires.add_tube_wire(tube_wire)  # add tube in wires

    # ---对所有线段进行切分----
   # wires.display()
    wires.split_long_wires_all(max_length)

    # 将表皮线段添加到空气线段集合中
    for tubeWire in wires.tube_wires:
        wires.add_air_wire(tubeWire.sheath)  # sheath wire is in the air, we need to calculate it in air part.
 #   wires.display()
    # 2. initialize ground
    ground_dic = tower_dict['ground']
    ground = initialize_ground(ground_dic)

    # 3. initialize lumps
    lumps,measurement = initial_lump(tower_dict['Lump'], dt, T,measurement)

    # 4. initialize devices
    devices,measurement = initial_device(tower_dict['Device'], dt, T,measurement)

    # 5. information of tower
    tower_info = tower_dict['Info']
    info = TowerInfo(tower_info['name'],tower_info['id'],tower_info['type'],tower_info['position'],tower_info['Vclass'],
         tower_info['Theta'],tower_info['mode_con'],tower_info['mode_gnd'], tower_info['pole_height'], tower_info['pole_head'])

    # 6. initalize tower
    tower = Tower(tower_dict['name'], info, wires, tube_wire, lumps, ground, devices, measurement)
    print(f"Tower:{tower.name} loaded.")
    return tower

def initialize_OHL(OHL_dict, max_length):


    # 1. initialize wires
    wires = Wires()
    for wire in OHL_dict['Wire']:
        #  initialize air wire
        wire_air = initialize_OHL_wire(wire)
        wire_air.start_node.x = wire_air.start_node.x + OHL_dict['Info']['Tower_head_pos'][0]
        wire_air.start_node.y = wire_air.start_node.y + OHL_dict['Info']['Tower_head_pos'][1]
        wire_air.start_node.z = wire_air.start_node.z + OHL_dict['Info']['Tower_head_pos'][2]

        wire_air.end_node.x = wire_air.end_node.x + OHL_dict['Info']['Tower_tail_pos'][0]
        wire_air.end_node.y = wire_air.end_node.y + OHL_dict['Info']['Tower_tail_pos'][1]
        wire_air.end_node.z = wire_air.end_node.z + OHL_dict['Info']['Tower_tail_pos'][2]

        wires.add_air_wire(wire_air)  # add air wire in wires

   # wires.display()

    # 2. initialize ground
    ground_dic = OHL_dict['ground']
    ground = initialize_ground(ground_dic)
    # 3. initialize info
    OHL_info = OHL_dict['Info']
    info = OHLInfo(OHL_info['name'],OHL_info['id'],OHL_info['type'],OHL_info['dL'],OHL_info['model1'],
         OHL_info['model2'],OHL_info['Tower_head'],OHL_info['Tower_head_id'], OHL_info['Tower_head_pos'],
                     OHL_info['Tower_tail'],OHL_info['Tower_tail_id'],  OHL_info['Tower_tail_pos'])

    # 4. initalize ohl
    ohl = OHL(OHL_info['name'], info, wires, None, len(OHL_dict['Wire']), ground)
   # ohl.wires_name = list(ohl.wires.get_all_wires().keys())
   # ohl.nodes_name = ohl.wires.get_all_nodes()
    print(f"OHL:{ohl.info.name} loaded.")
    return ohl

def initialize_measurement(file_name):
    json_file_path = "Data/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r') as j:
        load_dict = json.load(j)

def initial_lump(lump_data, dt, T,measurement):
    Nt = int(np.ceil(T / dt))

    lumps = Lumps()

    nolinear_element_parameters = Nolinear_Element_Parameters()
    switch_parameters = Switch_Parameters()
    for lump in lump_data:
        lump_type = lump['Type']
        name = lump['name']
        bran_name = np.array([lump['bran_name']]).reshape(-1)
        node1 = np.array([lump['node1']]).reshape(-1)
        node2 = np.array([lump['node2']]).reshape(-1)
        label = 1
        if lump['probe']:
            measurement[lump['name']]= [label,lump['probe'],bran_name,node1,node2]
        # 判断器件类型
        match lump_type:
            case 'RL':
                resistance = lump['value1']
                inductance = lump['value2']
                lumps.add_resistor_inductor(
                    Resistor_Inductor(name, bran_name, node1, node2, resistance, inductance))
            case 'GC':
                conductance = lump['value1']
                capacitance = lump['value2']
                lumps.add_conductor_capacitor(
                    Conductor_Capacitor(name, bran_name, node1, node2,  conductance, capacitance))
            case 'Vs':
                type_of_data = lump['data_type']
                resistance = lump['value1']
                if type_of_data > 0:
                    magnitude = lump['value2']
                    frequency = lump['value3']
                    angle = lump['value5']
                    lumps.add_voltage_source_cosine(
                        Voltage_Source_Cosine(name, bran_name, node1, node2, resistance, magnitude, frequency,
                                              angle))
                elif type_of_data == 0:
                    voltages = np.array(lump['value2'])
                    lumps.add_voltage_source_empirical(
                        Voltage_Source_Empirical(name, bran_name, node1, node2,  resistance, voltages))
            case 'Is':
                type_of_data = lump['data_type']
                if type_of_data > 0:
                    magnitude = lump['value2']
                    frequency = lump['value3']
                    angle = lump['value5']
                    lumps.add_current_source_cosine(
                        Current_Source_Cosine(name, bran_name, node1, node2, magnitude, frequency, angle))
                elif type_of_data == 0:
                    currents = np.array(lump['value2'])
                    lumps.add_current_source_empirical(
                        Current_Source_Empirical(name, bran_name, node1, node2, currents))
            case 'VCVS':
                #---mei zhilu，node1, node2电压-》支路电压
                resistance = lump['value1'][0]
                gain = lump['value1'][1]
                lumps.add_voltage_control_voltage_source(
                    Voltage_Control_Voltage_Source(name, bran_name, node1, node2, resistance, gain))
                if lump['probe']:
                    measurement[lump['name']] = [2, lump['probe'], bran_name, node1, node2,resistance,gain]
            case 'ICVS':
                # mei dian
                resistance = lump['value1'][0]
                r = lump['value1'][1]
                lumps.add_current_control_voltage_source(
                    Current_Control_Voltage_Source(name, bran_name, node1, node2, resistance, r))
                if lump['probe']:
                    measurement[lump['name']] = [3, lump['probe'], bran_name, node1, node2, resistance, r]
            case 'VCIS':
                # node1,2 电压——》电流，g
                g = lump['value1'][1]
                lumps.add_voltage_control_current_source(
                    Voltage_Control_Current_Source(name, bran_name, node1, node2, g))
                if lump['probe']:
                    measurement[lump['name']] = [4, lump['probe'], bran_name, node1, node2, g]
            case 'ICIS':
                # bran_name电流--》node电流，直接取节点电压
                gain = lump['value1'][1]
                lumps.add_current_control_current_source(
                    Current_Control_Current_Source(name, bran_name, node1, node2, gain))
                if lump['probe']:
                    measurement[lump['name']] = [5, lump['probe'], bran_name, node1, node2, gain]
            case 'TX2':
                vpri = lump['value1'][0]
                vsec = lump['value2'][0]
                lumps.add_transformer_one_phase(
                    Transformer_One_Phase(name, bran_name, node1, node2, vpri, vsec))
            case 'TX3':
                vpri = lump['value1'][0]
                vsec = lump['value2'][0]
                lumps.add_transformer_three_phase(
                    Transformer_Three_Phase(name, bran_name, node1, node2, vpri, vsec))
            case 'M2':
                resistance = lump['value1']
                inductance = np.array(lump['value1'][0])
                lumps.add_mutual_inductor_two_port(
                    Mutual_Inductance_Two_Port(name, bran_name, node1, node2, resistance, inductance))
            case 'M3':
                resistance = lump['value1']

                inductance = np.array(lump['value2'][0])
                lumps.add_mutual_inductor_three_port(
                    Mutual_Inductance_Three_Port(name, bran_name, node1, node2, resistance, inductance))
            case 'NLR':
                resistance = lump['value1']
                model = lump['model']
                if model != None:
                    nolinear_model = eval('nolinear_element_parameters.' + str(model))
                    vi_characteristics = nolinear_model['vi_characteristics']
                    ri_characteristics = nolinear_model['ri_characteristics']
                else:
                    vi_characteristics = np.array(lump['value2'])
                    ri_characteristics = np.array(lump['value3'])

                type_of_data = lump['data_type']
                lumps.add_nolinear_resistor(
                    Nolinear_Resistor(name, bran_name, node1, node2, resistance, vi_characteristics, ri_characteristics,
                                      type_of_data))
            case 'NLF':
                inductance = lump['value2']
                bh_characteristic = np.array(lump['value3'])
                type_of_data = lump['data_type']
                lumps.add_nolinear_f(
                    Nolinear_F(name, bran_name, node1, node2, inductance, bh_characteristic, type_of_data))
            case 'SWV':
                resistance = lump['value1']
                voltage = lump['value3']
                type_of_data = lump['data_type']
                lumps.add_voltage_controled_switch(
                    Voltage_Controled_Switch(name, bran_name, node1, node2, resistance, voltage, type_of_data))
            case 'SWT':
                close_time = lump['value1']
                open_time = lump['value2']
                type_of_data = lump['data_type']
                lumps.add_time_controled_switch(
                    Time_Controled_Switch(name, bran_name, node1, node2, close_time, open_time, type_of_data))
            case 'A2G':
                resistance = lump['value1']
                lumps.add_a2g(
                    A2G(name, bran_name, node1, node2, resistance))
            case 'ROD':
                resistance = lump['value1']
                inductance = lump['value2']
                lumps.add_a2g(
                    Ground(name, bran_name, node1, node2, resistance, inductance))
            case "swh":
                resistance = lump['value1']
                model = lump['model']
                if model != None:
                    nolinear_model = eval('switch_parameters.' + str(model))
                    DE_max = nolinear_model['DE_max']
                    v_initial = nolinear_model['v_initial']
                    k = nolinear_model['k']
                else:
                    DE_max = lump['value2']
                    v_initial = lump['value3']
                    k = lump['value4']
                lumps.add_switch_disruptive_effect_model(
                    Switch_Disruptive_Effect_Model(name, bran_name, node1, node2, resistance, 0, DE_max, v_initial, k))
            case 'MTCK':
                dist = np.array(lump['distance'])
                high = np.array(lump['high'])
                radius = np.array(lump['r0'])



    # 获取器件不重复的支路列表和节点列表
    lumps.brans_nodes_list_initial()
    # 初始化Lumps参数矩阵
    lumps.lump_parameter_matrix_initial()
    # 将器件的参数分配到参数矩阵中
    lumps.parameters_assign()
    # 初始化Lumps电压矩阵
    lumps.lump_voltage_source_matrix_initial(T, dt)
    # 初始化Lumps电流矩阵
    lumps.lump_current_source_matrix_initial(T, dt)
    return lumps,measurement

def initial_device(device_data, dt, T,measurement):
    devices = Devices()

    for device in device_data:
        lumps,measurement = initial_lump(device['Lump'], dt, T,measurement)
        if device['type'] == 'insulator':
            devices.add_insolator(lumps)
        elif device['type'] == 'arrestor':
            devices.add_arrestor(lumps)
        elif device['type'] == 'transformer':
            devices.add_transformer(lumps)
    # d = data.shape[0]

    return devices,measurement

def initial_source(network, nodes, load_dict,dt):
    # 0. read json file
    stroke_list = []
    for stroke_dict in load_dict['Stroke']:
        duration,stroke_type,parameters,parameter_set= stroke_dict['duration'],stroke_dict['type'],stroke_dict['parameters'],stroke_dict['parameter_set']
        stroke = Stroke(stroke_type, duration=duration, dt=dt, is_calculated=True, parameter_set=parameter_set, parameters=None)
        stroke.calculate()
        stroke_list.append(stroke)
    lightning =Lightning(id=1, type= load_dict['type'], strokes=stroke_list, channel=Channel(load_dict['position']))
    branches = segment_branch(network.branches)
    start = [list(l[0].values())[0] for l in list(branches.values())]
    end = [list(l[1].values())[0] for l in list(branches.values())]
    branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
    pt_start = np.array(start)
    pt_end = np.array(end)
    constants = Constant()
    # constants.ep0 = 8.85e-12
    U_out = pd.DataFrame()
    I_out = pd.DataFrame()
    for i in range(len(stroke_list)):
        U_out = pd.concat([U_out, InducedVoltage_calculate(pt_start, pt_end, branches, lightning,
                            stroke_sequence=i, constants=constants)], axis=1,ignore_index=True)
        I_out = pd.concat([I_out, LightningCurrent_calculate(load_dict["area"], load_dict["wire"], load_dict["position"],
                            network, nodes, lightning, stroke_sequence=i)],axis=1,ignore_index=True)
   # Source_Matrix = pd.concat([I_out, U_out], axis=0)
    for model_list in [network.towers, network.OHLs, network.cables]:
        for model in model_list:
            U_out = U_out.add(model.voltage_source_matrix, fill_value=0).fillna(0)
            I_out = I_out.add(model.current_source_matrix, fill_value=0).fillna(0)
    Source_Matrix = pd.concat([U_out, I_out], axis=0)
    return Source_Matrix

def initialize_cable(cable, max_length,VF):

    # 0. initialize info
    cable_info = cable['Info']
    info = CableInfo(cable_info['name'],cable_info['id'],cable_info['type'],cable_info['T_head'],cable_info['T_head_id'],
    cable_info['T_head_pos'],cable_info['T_tail'], cable_info['T_tail_id'],cable_info['T_tail_pos'],
    cable_info['core_num'],cable_info['armor_num'],cable_info['delta_L'], cable_info['mode_con'], cable_info['mode_gnd'])

    # 1. initialize wires
    wire = cable
    wires = Wires()
    nodes = []
    sheath_wire = initialize_wire(wire['TubeWire']['sheath'], nodes,VF)
    sheath_wire.start_node.x = sheath_wire.start_node.x+ cable['Info']['T_head_pos'][0]
    sheath_wire.start_node.y = sheath_wire.start_node.y+ cable['Info']['T_head_pos'][1]
    sheath_wire.start_node.z = sheath_wire.start_node.z+ cable['Info']['T_head_pos'][2]

    sheath_wire.end_node.x = sheath_wire.end_node.x + cable['Info']['T_tail_pos'][0]
    sheath_wire.end_node.y = sheath_wire.end_node.y + cable['Info']['T_tail_pos'][1]
    sheath_wire.end_node.z = sheath_wire.end_node.z + cable['Info']['T_tail_pos'][2]
    tube_wire = TubeWire(sheath_wire, wire['TubeWire']['sheath']['rs1'], wire['TubeWire']['sheath']['rs3'],
                         wire['TubeWire']['sheath']['core_num'])


    for core in wire['TubeWire']['core']:
        core_wire = initialize_wire(core, nodes,VF)
        core_wire.start_node.x = core_wire.start_node.x +cable['Info']['T_head_pos'][0]
        core_wire.start_node.y = core_wire.start_node.x + cable['Info']['T_head_pos'][1]
        core_wire.start_node.z = core_wire.start_node.x + cable['Info']['T_head_pos'][2]

        core_wire.end_node.x = core_wire.end_node.x +cable['Info']['T_tail_pos'][0]
        core_wire.end_node.y = core_wire.end_node.x + cable['Info']['T_tail_pos'][1]
        core_wire.end_node.z = core_wire.end_node.x + cable['Info']['T_tail_pos'][2]

        tube_wire.add_core_wire(core_wire)

    wires.add_tube_wire(tube_wire)

   # wires.display()
    wires.split_long_wires_all(max_length)

    # 2. initialize ground
    ground_dic = cable['ground']
    ground = initialize_ground(ground_dic)

    # 3. initalize cable
    cable = Cable(cable['name'], info, wires, ground)
    cable.wires_name = cable.wires.get_all_wires()
    cable.nodes_name = cable.wires.get_all_nodes()
    print("Cable loaded.")
    return cable

def print_lumps(lumps):

    matrix_A = lumps.incidence_matrix_A
    print("incidence_matrix_A",matrix_A)

    matrix_B = lumps.incidence_matrix_B
    print('incidence_matrix_B ?',matrix_B)

    resistance_matrix = lumps.resistance_matrix
    print('resistance_matrix ?',resistance_matrix)

    inductance_matrix = lumps.inductance_matrix
    print('inductance_matrix ', inductance_matrix)

    conductance_matrix = lumps.conductance_matrix
    print('conductance_matrix equals?',conductance_matrix)

    capacitance_matrix = lumps.capacitance_matrix
    print('capacitance_matrix equals?',capacitance_matrix)

# if __name__ == '__main__':
#     file_name = "01_2"
#     json_file_path = "../../Data/" + file_name + ".json"
#     # 0. read json file
#     with open(json_file_path, 'r') as j:
#         load_dict = json.load(j)
#
#     # 1. initialize all elements in the network
#     cable = initialize_cable(load_dict['Cable'][0],50)
#     print(cable.info.HeadTower)


# def initialize_measurement(file_name):
#
