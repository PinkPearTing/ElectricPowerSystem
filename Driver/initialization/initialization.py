import json
import numpy as np

from Model.Lump import Lumps, Resistor_Inductor, Measurement_Linear, Conductor_Capacitor, Measurement_GC, \
    Voltage_Source_Cosine, Voltage_Source_Empirical, Current_Source_Cosine, Str2martix, Current_Source_Empirical, \
    Voltage_Control_Voltage_Source, Current_Control_Voltage_Source, Voltage_Control_Current_Source, \
    Current_Control_Current_Source, Transformer_One_Phase, Transformer_Three_Phase, Mutual_Inductance_Two_Port, \
    Mutual_Inductance_Three_Port, Nolinear_Resistor, Nolinear_F, Voltage_Controled_Switch, Time_Controled_Switch, A2G
from Model.Node import Node
from Model.Wires import Wire, Wires, CoreWire, TubeWire
from Model.Ground import Ground
from Model.Tower import Tower
from Model.OHL import OHL
from Model.Lightning import Stroke
from Model.Contant import Constant
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate, Current_source_generate
import pandas as pd
from Model.Cable import Cable



# initialize wire in tower
def initialize_wire(wire, nodes):
    bran = wire['bran']
    node_name_start = wire['node1']
    pos_start = wire['pos_1']
    node_name_end = wire['node2']
    pos_end = wire['pos_2']
    node_start = Node(name=node_name_start, x=pos_start[0], y=pos_start[1], z=pos_start[2])
    node_end = Node(name=node_name_end, x=pos_end[0], y=pos_end[1], z=pos_end[2])


    offset = wire['oft']
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
    if wire['type'] == 'air' or wire['type'] == 'sheath' or wire['type'] == 'ground':
        return Wire(bran, node_start, node_end, offset, radius, R, L, sig, mur, epr, VF)
    elif wire['type'] == 'core':
        return CoreWire(bran, node_start, node_end, offset, radius, R, L, sig, mur, epr, VF, wire['rs2'], wire['rs3'])

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


def initialize_tower(file_name, max_length):
    json_file_path = "Data/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r') as j:
        load_dict = json.load(j)

    # 1. initialize wires
    wires = Wires()
    tube_wire = TubeWire(None, None, None, None)
    nodes = []
    for wire in load_dict['Tower']['Wire']:

        # 1.1 initialize air wire
        if wire['type'] == 'air':
            wire_air = initialize_wire(wire, nodes)
            wires.add_air_wire(wire_air)  # add air wire in wires

        # 1.2 initialize ground wire
        elif wire['type'] == 'ground':
            wire_ground = initialize_wire(wire, nodes)
            wires.add_ground_wire(wire_ground)  # add ground wire in wires

        # 1.3 initialize tube
        elif wire['type'] == 'tube':
            sheath_wire = initialize_wire(wire['sheath'], nodes)
            tube_wire = TubeWire(sheath_wire, wire['sheath']['rs2'], wire['sheath']['rs3'], wire['sheath']['num'])

            for core in wire['core']:
                core_wire = initialize_wire(core, nodes)
                tube_wire.add_core_wire(core_wire)

            wires.add_tube_wire(tube_wire)  # add tube in wires

    # ---对所有线段进行切分----
    wires.display()
    wires.split_long_wires_all(max_length)

    # 将表皮线段添加到空气线段集合中
    for tubeWire in wires.tube_wires:
        wires.add_air_wire(tubeWire.sheath)  # sheath wire is in the air, we need to calculate it in air part.
    wires.display()
    # 2. initialize ground
    ground_dic = load_dict['Tower']['ground']
    ground = initialize_ground(ground_dic)

    # 3. initialize lumps
    lumps = initial_lump(load_dict['Tower']['Lump'])

    # 4. initalize tower
    tower = Tower(None, wires, tube_wire, lumps, ground, None, None)
    print("Tower loaded.")
    return tower

def initialize_OHL(file_name, max_length):

    json_file_path = "Data/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r') as j:
        load_dict = json.load(j)

    # 1. initialize wires
    wires = Wires()
    for wire in load_dict['OHL']['Wire']:

        #  initialize air wire
            wire_air = initialize_OHL_wire(wire)
            wires.add_air_wire(wire_air)  # add air wire in wires

    wires.display()

    # 2. initialize ground
    ground_dic = load_dict['OHL']['ground']
    ground = initialize_ground(ground_dic)

    # 3. initalize ohl
    ohl = OHL(None, None, wires, None, None, ground)
    print("OHL loaded.")
    return ohl

def initialize_measurement(file_name):
    json_file_path = "Data/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r') as j:
        load_dict = json.load(j)

def initial_lump(lump_data):
    dt = 1e-6
    T = 0.001
    Nt = int(np.ceil(T / dt))
    # Nbran = Bran['num'][0]
    # Nnode = Node['num'][0]
    # nodebran_n = np.copy(data[:, 2:5])
    # nodebran_id = NodeBranIndex_Update(Node, Bran, data[:, 2:5])

    lumps = Lumps()
    # d = data.shape[0]
    for lump in lump_data:
        lump_type = lump['Type']
        name = lump['name']
        bran_name = lump['bran_name']
        node1= lump['node1']
        node2= lump['node2']
        match lump_type:
            case 'RL':
                resistance = lump['value1']
                inductance = lump['value2']
                lumps.add_resistor_inductor(
                    Resistor_Inductor(name, bran_name, node1, node2, resistance, inductance))
                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'GC':
                conductance = lump['value1']
                capacitance = lump['value2']
                lumps.add_conductor_capacitor(
                    Conductor_Capacitor(name, bran_name, node1, node2,  conductance, capacitance))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_gc(
                        Measurement_GC(name, bran_name, node1, node2,  probe, conductance, capacitance))
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
                    voltages = Str2martix(lump['value2'])
                    lumps.add_voltage_source_empirical(
                        Voltage_Source_Empirical(name, bran_name, node1, node2,  resistance, voltages))

                probe = ['prob']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'Is':
                type_of_data = lump['data_type']
                if type_of_data > 0:
                    magnitude = lump['value2']
                    frequency = lump['value3']
                    angle = lump['value5']
                    lumps.add_current_source_cosine(
                        Current_Source_Cosine(name, bran_name, node1, node2, magnitude, frequency, angle))
                elif type_of_data == 0:
                    currents = Str2martix(lump['value2'])
                    lumps.add_current_source_empirical(
                        Current_Source_Empirical(name, bran_name, node1, node2, currents))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'VCVS':
                resistance = lump['value1'][0]
                gain = lump['value1'][1]
                lumps.add_voltage_control_voltage_source(
                    Voltage_Control_Voltage_Source(name, bran_name, node1, node2, resistance, gain))
                probe = lump['probe']
                for ith_probe in range(2):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name,bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe[ith_probe]))
            case 'ICVS':
                resistance = lump['value1'][0]
                r = lump['value1'][1]
                lumps.add_current_control_voltage_source(
                    Current_Control_Voltage_Source(name, bran_name, node1, node2, resistance, r))
                probe = lump['probe']

                for ith_probe in range(2):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe],
                                               node1[ith_probe], node2[ith_probe], probe[ith_probe]))
            case 'VCIS':
                g = lump['value1'][1]
                lumps.add_voltage_control_current_source(
                    Voltage_Control_Current_Source(name, bran_name, node1, node2, g))
                probe = lump['probe']
                for ith_probe in range(2):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe))
            case 'ICIS':
                gain = lump['value1'][1]
                lumps.add_current_control_current_source(
                    Current_Control_Current_Source(name, bran_name, node1, node2, gain))
                probe = lump['probe']
                for ith_probe in range(2):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe))
            case 'TX2':
                vpri = lump['value1'][0]
                vsec = lump['value2'][0]
                lumps.add_transformer_one_phase(
                    Transformer_One_Phase(name, bran_name, node1, node2, vpri, vsec))
                probe = lump['probe']
                for ith_probe in range(2):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe))
            case 'TX3':

                vpri = lump['value1'][0]
                vsec = lump['value2'][0]
                lumps.add_transformer_three_phase(
                    Transformer_Three_Phase(name, bran_name, node1, node2, vpri, vsec))
                probe = lump['probe']
                for ith_probe in range(6):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe))
            case 'M2':
                resistance = lump['value1']
                inductance = Str2martix(lump['value1'][0])
                lumps.add_mutual_inductor_two_port(
                    Mutual_Inductance_Two_Port(name, bran_name, node1, node2, resistance, inductance))
                probe = lump['probe']
                for ith_probe in range(2):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe))
            case 'M3':
                resistance = lump['value1']

                inductance = Str2martix(lump['value2'][0])
                lumps.add_mutual_inductor_three_port(
                    Mutual_Inductance_Three_Port(name, bran_name, node1, node2, resistance, inductance))
                probe = lump['probe']
                for ith_probe in range(3):
                    if str(probe) != 'nan':
                        lumps.measurements.add_measurement_linear(
                            Measurement_Linear(name + '_{}'.format(ith_probe), bran_name[ith_probe], node1[ith_probe],
                                               node2[ith_probe], probe))
            case 'NLR':

                resistance = lump['value1']
                vi_characteristic = Str2martix(lump['value3'])
                type_of_data = lump['data_type']
                lumps.add_nolinear_resistor(
                    Nolinear_Resistor(name, bran_name, node1, node2, resistance, vi_characteristic, type_of_data))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'NLF':
                inductance = lump['value2']
                bh_characteristic = Str2martix(lump['value3'])
                type_of_data = lump['data_type']
                lumps.add_nolinear_f(
                    Nolinear_F(name, bran_name, node1, node2, inductance, bh_characteristic, type_of_data))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'SWV':

                resistance = lump['value1']
                voltage = lump['value3']
                type_of_data = lump['data_type']
                lumps.add_voltage_controled_switch(
                    Voltage_Controled_Switch(name, bran_name, node1, node2, resistance, voltage, type_of_data))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'SWT':
                close_time = lump['value1']
                open_time = lump['value2']
                type_of_data = lump['data_type']
                lumps.add_time_controled_switch(
                    Time_Controled_Switch(name, bran_name, node1, node2, close_time, open_time, type_of_data))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'A2G':
                resistance = lump['value1']
                lumps.add_a2g(
                    A2G(name, bran_name, node1, node2, resistance))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))
            case 'GOD':
                resistance = lump['value1']
                inductance = lump['value2']
                lumps.add_a2g(
                    Ground(name, bran_name, node1, node2, resistance, inductance))

                probe = lump['probe']
                if str(probe) != 'nan':
                    lumps.measurements.add_measurement_linear(
                        Measurement_Linear(name, bran_name, node1, node2, probe))

    lumps.brans_nodes_list_initial()
    lumps.lump_parameter_martix_initial()
    lumps.lump_measurement_initial(Nt)
    lumps.parameters_assign()
    lumps.lump_voltage_source_martix_initial(T, dt)
    lumps.lump_current_source_martix_initial(T, dt)
    print_lumps(lumps)
    return lumps

def initial_source(network, nodes, file_name):
    json_file_path = "Data/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r') as j:
        load_dict = json.load(j)

    stroke = Stroke('Heidler', duration=0.1, is_calculated=True, hit_pos=[500, 50, 0], parameter_set=None, parameters=None)
    pt_start = np.array(network.starts)
    pt_end = np.array(network.ends)
    i_sr = pd.read_excel('i_sr.xlsx', header=None)
    i_sr = i_sr.to_numpy()
    stroke.current_waveform = i_sr
    constants = Constant()
    constants.ep0 = 8.85e-12

    U_out = InducedVoltage_calculate(pt_start, pt_end, stroke, constants)
    I_out = Current_source_generate(load_dict["Source"]["area"],load_dict["Source"]["wire"],load_dict["Source"]["position"],network, nodes)



def print_lumps(lumps):

    matrix_A = lumps.incidence_matrix_A
    print("incidence_matrix_A",matrix_A)

    matrix_B = lumps.incidence_matrix_B
    print('incidence_matrix_B ?',matrix_B)

    resistance_matrix = lumps.resistance_matrix
    print('resistance_matrix ?',resistance_matrix)

    inductance_matrix = lumps.inductance_matrix
    print('inductance_matrix ', inductance_matrix)

    conductance_martix = lumps.conductance_martix
    print('conductance_martix equals?',conductance_martix)

    capacitance_martix = lumps.capacitance_martix
    print('capacitance_martix equals?',capacitance_martix)




# def initialize_measurement(file_name):
#
