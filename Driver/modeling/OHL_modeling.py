import numpy as np
import pandas as pd

from Function.Calculators.Impedance import calculate_OHL_impedance
from Function.Calculators.Capacitance import calculate_OHL_capcitance
from Function.Calculators.Inductance import calculate_OHL_mutual_inductance, calculate_OHL_inductance
from Function.Calculators.Resistance import calculate_OHL_resistance
from Model.Contant import Constant
from Utils.Math import distance
# A矩阵
def build_incidence_matrix(OHL, segment_num):
    # A矩阵
    print("------------------------------------------------")
    print("A_OHL matrix is building...")
    # 初始化A矩阵
    incidence_martix = np.zeros((len(OHL.wires_name), len(OHL.nodes_name)))
    wires_num = OHL.phase_num

    for i in range(segment_num*wires_num):
        incidence_martix[i, i] = -1
        incidence_martix[i, i+wires_num] = 1

    OHL.incidence_matrix = pd.DataFrame(incidence_martix, index=OHL.wires_name, columns=OHL.nodes_name)

   # print(OHL.incidence_matrix)
    print("A_OHL matrix is built successfully")
    print("------------------------------------------------")
# R矩阵
def build_resistance_matrix(OHL, segment_num, segment_length):
    # R矩阵
    print("------------------------------------------------")
    print("R matrix is building...")
    wires_num = OHL.phase_num
    R = calculate_OHL_resistance(OHL.wires.get_resistance())
    resistance_martix = np.zeros((len(OHL.wires_name), len(OHL.wires_name)))

    for i in range(segment_num):
        resistance_martix[i*wires_num:(i+1)*wires_num, i*wires_num:(i+1)*wires_num] = R * segment_length

    OHL.resistance_matrix = pd.DataFrame(resistance_martix, index=OHL.wires_name, columns=OHL.wires_name, dtype=float)

 #   print(OHL.resistance_matrix)
    print("R_OHL matrix is built successfully")
    print("------------------------------------------------")
# L矩阵
def build_inductance_matrix(OHL, Lm, segment_num, segment_length):
    # L矩阵
    print("------------------------------------------------")
    print("L_OHL matrix is building...")
    wires_num = OHL.wires.count()
    L = calculate_OHL_inductance(OHL.wires.get_inductance(), Lm)

    inductance_martix = np.zeros((len(OHL.wires_name), len(OHL.wires_name)))

    for i in range(segment_num):
        inductance_martix[i * wires_num:(i + 1) * wires_num, i * wires_num:(i + 1) * wires_num] = L * segment_length

    OHL.inductance_matrix = pd.DataFrame(inductance_martix, index=OHL.wires_name, columns=OHL.wires_name, dtype=float)

   # print(OHL.inductance_matrix)
    print("L_OHL matrix is built successfully")
    print("------------------------------------------------")

# C矩阵
def build_capacitance_matrix(OHL, Lm, segment_num, segment_length, constant):
    # C矩阵
    print("------------------------------------------------")
    print("C_OHL matrix is building...")

    wires_num = OHL.wires.count()
    C = calculate_OHL_capcitance(Lm, constant)

    capacitance_martix = np.zeros((len(OHL.nodes_name), len(OHL.nodes_name)))

    for i in range(segment_num + 1):
        capacitance_martix[i * wires_num:(i + 1) * wires_num, i * wires_num:(i + 1) * wires_num] = 0.5 * C * segment_length if i == 0 or i == segment_num else C * segment_length
        # 与外界相连接的部分，需要折半

    OHL.capacitance_matrix = pd.DataFrame(capacitance_martix, index=OHL.nodes_name, columns=OHL.nodes_name, dtype=float)

 #   print(OHL.capacitance_matrix)
    print("C_OHL matrix is built successfully")
    print("------------------------------------------------")

# G矩阵
def build_conductance_matrix(OHL):
    # G矩阵
    print("------------------------------------------------")
    print("G_OHL matrix is building...")

    OHL.conductance_matrix = pd.DataFrame(0, index=OHL.nodes_name, columns=OHL.nodes_name, dtype=float)

   # print(OHL.conductance_matrix)
    print("G_OHL matrix is built successfully")
    print("------------------------------------------------")

# Z矩阵
def build_impedance_matrix(OHL, Lm, constants, frequency, ground):
    # Z矩阵
    print("------------------------------------------------")
    print("Z_OHL matrix is building...")
    OHL.impedance_martix = calculate_OHL_impedance(OHL.wires.get_radii(), OHL.wires.get_mur(), OHL.wires.get_sig(), OHL.wires.get_epr(),
                                OHL.wires.get_offsets(), OHL.wires.get_heights(), ground.sig, ground.mur,
                                ground.epr, Lm, constants, frequency)

    print("Z_OHL matrix is built successfully")
    print("------------------------------------------------")

def OHL_building(OHL,  max_length, frequency, ground):
    print("------------------------------------------------")
    print(f"OHL:{OHL.info.name} building...")
    # 0.参数准备
    constants = Constant()
    OHL_r = OHL.wires.get_radii()
    OHL_height = OHL.wires.get_heights()
    length = distance(OHL.info.HeadTower_pos,OHL.info.TailTower_pos)

    segment_num = int(np.ceil(length / max_length))
    segment_length = length/segment_num

    Lm = calculate_OHL_mutual_inductance(OHL_r, OHL_height, OHL.wires.get_offsets(), constants)

    OHL.get_brans_nodes_list(segment_num)

    # 1. 构建A矩阵
    build_incidence_matrix(OHL, segment_num)

    # 2. 构建R矩阵
    build_resistance_matrix(OHL, segment_num, segment_length)

    # 3. 构建L矩阵
    build_inductance_matrix(OHL, Lm, segment_num, segment_length)

    # 4. 构建C矩阵
    build_capacitance_matrix(OHL, Lm, segment_num, segment_length, constants)

    # 5. 构建G矩阵
    build_conductance_matrix(OHL)

    # 6. 构建Z矩阵
    build_impedance_matrix(OHL, Lm, constants, frequency, ground)
    print(f"OHL:{OHL.info.name} building is completed.")
    print("------------------------------------------------")
