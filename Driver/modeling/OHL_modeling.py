import numpy as np
import pandas as pd

from Function.Calculators.Impedance import calculate_OHL_impedance, calculate_OHL_wire_impedance, calculate_OHL_ground_impedance
from Function.Calculators.Capacitance import calculate_OHL_capcitance
from Function.Calculators.Inductance import calculate_OHL_mutual_inductance, calculate_OHL_inductance
from Function.Calculators.Resistance import calculate_OHL_resistance
from Model.Contant import Constant
from Utils.Math import distance
from Vector_Fitting.Drivers.MatrixFitting import matrix_vector_fitting
from Driver.modeling_varient_freqency.recursive_convolution import preparing_parameters

# A矩阵
def build_incidence_matrix(OHL, segment_num):
    # A矩阵
    print("------------------------------------------------")
    print("A_OHL matrix is building...")
    # 初始化A矩阵
    incidence_matrix = np.zeros((len(OHL.wires_name), len(OHL.nodes_name)))
    wires_num = OHL.phase_num

    for i in range(segment_num*wires_num):
        incidence_matrix[i, i] = -1
        incidence_matrix[i, i+wires_num] = 1

    OHL.incidence_matrix = pd.DataFrame(incidence_matrix, index=OHL.wires_name, columns=OHL.nodes_name)

    if 'ref' in OHL.nodes_name:
        OHL.incidence_matrix.drop('ref', axis=1, inplace=True)

   # print(OHL.incidence_matrix)
    print("A_OHL matrix is built successfully")
    print("------------------------------------------------")
# R矩阵
def build_resistance_matrix(OHL, R, segment_num, segment_length):
    # R矩阵
    print("------------------------------------------------")
    print("R matrix is building...")
    wires_num = OHL.phase_num
    resistance_matrix = np.zeros((len(OHL.wires_name), len(OHL.wires_name)))

    for i in range(segment_num):
        resistance_matrix[i*wires_num:(i+1)*wires_num, i*wires_num:(i+1)*wires_num] = R * segment_length

    OHL.resistance_matrix = pd.DataFrame(resistance_matrix, index=OHL.wires_name, columns=OHL.wires_name, dtype=float)

 #   print(OHL.resistance_matrix)
    print("R_OHL matrix is built successfully")
    print("------------------------------------------------")
# L矩阵
def build_inductance_matrix(OHL, L, segment_num, segment_length):
    # L矩阵
    print("------------------------------------------------")
    print("L_OHL matrix is building...")
    wires_num = OHL.wires.count()

    inductance_matrix = np.zeros((len(OHL.wires_name), len(OHL.wires_name)))

    for i in range(segment_num):
        inductance_matrix[i * wires_num:(i + 1) * wires_num, i * wires_num:(i + 1) * wires_num] = L * segment_length

    OHL.inductance_matrix = pd.DataFrame(inductance_matrix, index=OHL.wires_name, columns=OHL.wires_name, dtype=float)

   # print(OHL.inductance_matrix)
    print("L_OHL matrix is built successfully")
    print("------------------------------------------------")

# C矩阵
def build_capacitance_matrix(OHL, C, segment_num, segment_length):
    # C矩阵
    print("------------------------------------------------")
    print("C_OHL matrix is building...")

    wires_num = OHL.wires.count()

    capacitance_matrix = np.zeros((len(OHL.nodes_name), len(OHL.nodes_name)))

    for i in range(segment_num + 1):
        capacitance_matrix[i * wires_num:(i + 1) * wires_num, i * wires_num:(i + 1) * wires_num] = 0.5 * C * segment_length if i == 0 or i == segment_num else C * segment_length
        # 与外界相连接的部分，需要折半

    OHL.capacitance_matrix = pd.DataFrame(capacitance_matrix, index=OHL.nodes_name, columns=OHL.nodes_name, dtype=float)

    if 'ref' in OHL.nodes_name:
        OHL.capacitance_matrix.drop('ref', axis=0, inplace=True)
        OHL.capacitance_matrix.drop('ref', axis=1, inplace=True)

 #   print(OHL.capacitance_matrix)
    print("C_OHL matrix is built successfully")
    print("------------------------------------------------")

# G矩阵
def build_conductance_matrix(OHL):
    # G矩阵
    print("------------------------------------------------")
    print("G_OHL matrix is building...")

    OHL.conductance_matrix = pd.DataFrame(0, index=OHL.nodes_name, columns=OHL.nodes_name, dtype=float)

    if 'ref' in OHL.nodes_name:
        OHL.conductance_matrix.drop('ref', axis=0, inplace=True)
        OHL.conductance_matrix.drop('ref', axis=1, inplace=True)

   # print(OHL.conductance_matrix)
    print("G_OHL matrix is built successfully")
    print("------------------------------------------------")

# Z矩阵
def build_impedance_matrix(OHL, Lm, constants, frequency, ground):
    # Z矩阵
    print("------------------------------------------------")
    print("Z_OHL matrix is building...")
    OHL.impedance_matrix = calculate_OHL_impedance(OHL.wires.get_radii(), OHL.wires.get_mur(), OHL.wires.get_sig(), OHL.wires.get_epr(),
                                OHL.wires.get_offsets(), OHL.wires.get_heights(), ground.sig, ground.mur,
                                ground.epr, Lm, constants, frequency)

    print("Z_OHL matrix is built successfully")
    print("------------------------------------------------")


def build_current_source_matrix(OHL, I):
    OHL.current_source_matrix = pd.DataFrame(I, index=OHL.nodes_name, columns=[0])

    if 'ref' in OHL.nodes_name:
        OHL.current_source_matrix.drop('ref', axis=0, inplace=True)



def build_voltage_source_matrix(OHL, V):
    # 获取线名称列表
    wire_name_list = OHL.wires_name
    OHL.voltage_source_matrix = pd.DataFrame(V, index=wire_name_list, columns=[0])

def prepare_building_parameters(OHL, max_length, ground, frequency, constants):
    OHL_r = OHL.wires.get_radii()
    OHL_height = OHL.wires.get_heights()
    length = distance(OHL.info.HeadTower_pos,OHL.info.TailTower_pos)

    segment_num = int(np.ceil(length / max_length))
    segment_length = length/segment_num

    Lm = calculate_OHL_mutual_inductance(OHL_r, OHL_height, OHL.wires.get_offsets(), constants.mu0)

    Zg = calculate_OHL_ground_impedance(ground.sig, ground.mur, ground.epr, OHL.wires.get_radii(),
                                        OHL.wires.get_offsets(), OHL.wires.get_heights(), constants,
                                        frequency).squeeze() if OHL.info.model2 != 0 and frequency != 0 else 0

    Lg = 0 if frequency == 0 else np.imag(Zg) / (2 * np.pi * frequency)

    resistance = calculate_OHL_resistance(OHL.wires.get_resistance(), OHL.wires.get_sig(), OHL.wires.get_mur(), OHL.wires.get_radii(), frequency, constants) + np.real(Zg)
    inductance = calculate_OHL_inductance(OHL.wires.get_inductance(), Lm, OHL.wires.get_sig(), OHL.wires.get_mur(), OHL.wires.get_radii(), frequency, constants) + Lg
    capcitance = calculate_OHL_capcitance(Lm, constants.mu0, constants.ep0)
    # R = pd.DataFrame(resistance)
    # R.to_excel('C:\\Users\\User\\Desktop\\SEMP_ohl_python_R_dis_{}Hz.xlsx'.format(frequency))
    # L = pd.DataFrame(inductance*(2 * np.pi * frequency))
    # L.to_excel('C:\\Users\\User\\Desktop\\SEMP_ohl_python_wL_dis_{}Hz.xlsx'.format(frequency))
    # C = pd.DataFrame(capcitance)
    # C.to_excel('C:\\Users\\User\\Desktop\\SEMP_ohl_python_C_dis_{}Hz.xlsx'.format(frequency))
    return segment_num, segment_length, resistance, capcitance, inductance

def build_OHL_parameter_matrixs(OHL, segment_num, segment_length, R, C, L):
    # 1. 构建A矩阵
    build_incidence_matrix(OHL, segment_num)

    # 2. 构建R矩阵
    build_resistance_matrix(OHL, R, segment_num, segment_length)

    # 3. 构建L矩阵
    build_inductance_matrix(OHL, L, segment_num, segment_length)

    # 4. 构建C矩阵
    build_capacitance_matrix(OHL, C, segment_num, segment_length)

    # 5. 构建G矩阵
    build_conductance_matrix(OHL)
    # build_conductance_matrix_jmarti(OHL, C, segment_num, segment_length)

    build_current_source_matrix(OHL, 0)

    build_voltage_source_matrix(OHL, 0)

def OHL_building(OHL,  max_length, gnd, frequency):
    print("------------------------------------------------")
    print(f"OHL:{OHL.info.name} building...")
    # 0.参数准备
    constants = Constant()

    segment_num, segment_length, R, C, L = prepare_building_parameters(OHL, max_length, gnd, frequency, constants)

    OHL.get_brans_nodes_list(segment_num)

    build_OHL_parameter_matrixs(OHL, segment_num, segment_length, R, C, L)

    print(f"OHL:{OHL.info.name} building is completed.")
    print("------------------------------------------------")


def prepare_variant_frequency_parameters(OHL, max_length, varied_frequency, Nfit, ground, dt, constants):
    OHL_r = OHL.wires.get_radii()
    OHL_height = OHL.wires.get_heights()
    length = distance(OHL.info.HeadTower_pos,OHL.info.TailTower_pos)

    segment_num = int(np.ceil(length / max_length))
    segment_length = length/segment_num

    OHL.get_brans_nodes_list(segment_num)

    Lm = calculate_OHL_mutual_inductance(OHL_r, OHL_height, OHL.wires.get_offsets(), constants)
    capacitance = calculate_OHL_capcitance(Lm, constants)

    if OHL.info.model1 == 1 and OHL.info.model2 == 2:
        Zc = calculate_OHL_wire_impedance(OHL.wires.get_radii(), OHL.wires.get_mur(), OHL.wires.get_sig(),
                                          OHL.wires.get_epr(), constants, varied_frequency)

        Zg = calculate_OHL_ground_impedance(ground.sig, ground.mur, ground.epr, OHL.wires.get_radii(),
                                            OHL.wires.get_offsets(), OHL.wires.get_heights(), constants,
                                            varied_frequency)

        Z = Zc + Zg

        SER = matrix_vector_fitting(Z, varied_frequency, Nfit)

        A, B = preparing_parameters(SER, dt)

        resistance = SER['D'] + A.sum(-1)

        inductance = SER['E'] + Lm

    elif OHL.info.model1 == 1:
        Zg = calculate_OHL_ground_impedance(ground.sig, ground.mur, ground.epr, OHL.wires.get_radii(),
                                            OHL.wires.get_offsets(), OHL.wires.get_heights(), constants,
                                            2e4) if OHL.info.model2 == 1 else 0

        Zc = calculate_OHL_wire_impedance(OHL.wires.get_radii(), OHL.wires.get_mur(), OHL.wires.get_sig(),
                                          OHL.wires.get_epr(), constants, varied_frequency)

        Ncon = OHL.phase_num

        resistance = np.zeros((Ncon, Ncon))
        inductance = np.zeros((Ncon, Ncon))

        A = np.zeros((Ncon, Ncon, Nfit))
        B = np.zeros((Ncon, 1, Nfit))

        for i in range(Ncon):
            SER = matrix_vector_fitting(Zc[i, i, :].reshape(1, 1, -1), varied_frequency, Nfit)

            SERA, SERB = preparing_parameters(SER, dt)

            A[i, i, :] = SERA
            B[i, 0, :] = SERB

            resistance[i, i] = np.real(SER['D'] + SERA.sum(-1))[0, 0]
            inductance[i, i] = np.real(SER['E'])[0]

        resistance += np.real(Zg).squeeze()
        inductance += Lm + np.imag(Zg).squeeze()/(2*np.pi*2e4)

    elif OHL.info.model2 == 2:
        Zg = calculate_OHL_ground_impedance(ground.sig, ground.mur, ground.epr, OHL.wires.get_radii(),
                                            OHL.wires.get_offsets(), OHL.wires.get_heights(), constants,
                                            varied_frequency)

        SER = matrix_vector_fitting(Zg, varied_frequency, Nfit)

        A, B = preparing_parameters(SER, dt)

        R = calculate_OHL_resistance(OHL.wires.get_resistance())

        L = calculate_OHL_inductance(OHL.wires.get_inductance(), Lm)

        resistance = SER['D'] + A.sum(-1) + R

        inductance = SER['E'] + L
    else:
        raise Exception('Conductor model and ground model of OHL are not variant frequency, since variant frequency model have used.')

    OHL.phi = np.zeros((len(OHL.wires_name), 1, Nfit))

    OHL.A = A * segment_length

    OHL.B = B

    return segment_num, segment_length, resistance, capacitance, inductance


def OHL_building_variant_frequency(OHL, max_length, ground, varied_frequency, Nfit, dt):
    print("------------------------------------------------")
    print(f"OHL:{OHL.info.name} building...")
    # 0.参数准备
    constants = Constant()

    segment_num, segment_length, R, C, L = prepare_variant_frequency_parameters(OHL, max_length, varied_frequency, Nfit, ground, dt, constants)

    build_OHL_parameter_matrixs(OHL, segment_num, segment_length, R, C, L)

    print(f"OHL:{OHL.info.name} building is completed.")
    print("------------------------------------------------")
