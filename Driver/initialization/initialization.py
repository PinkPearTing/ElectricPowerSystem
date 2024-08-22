import json
import numpy as np
from Model.Node import Node
from Model.Wires import Wire, Wires, CoreWire, TubeWire
from Model.Ground import Ground
from Model.Tower import Tower
from Model.OHL import OHL



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

    # 3. initalize tower
    tower = Tower(None, wires, tube_wire, None, ground, None, None)
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

    # 3. initalize tower
    ohl = OHL(None, None, wires, None, None, ground)
    print("OHL loaded.")
    return ohl

def initialize_measurement(file_name):
    json_file_path = "Data/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r') as j:
        load_dict = json.load(j)

