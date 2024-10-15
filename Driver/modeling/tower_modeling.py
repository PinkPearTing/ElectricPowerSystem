import numpy as np
from Function.Calculators.Inductance import calculate_coreWires_inductance, calculate_sheath_inductance, calculate_wires_inductance_potential_with_ground
from Function.Calculators.Capacitance import calculate_coreWires_capacitance, calculate_sheath_capacitance
from Function.Calculators.Impedance import calculate_coreWires_impedance, calculate_sheath_impedance, calculate_multual_impedance, calculate_OHL_wire_impedance
from Model.Contant import Constant
from Vector_Fitting.Drivers.MatrixFitting import matrix_vector_fitting
from Driver.modeling_varient_freqency.recursive_convolution import preparing_parameters
from Vector_Fitting.Calculators.plots import plot_figure_11

from scipy.linalg import block_diag
import pandas as pd

# A
def build_incidence_matrix(tower):
    # A矩阵
    print("------------------------------------------------")
    print("A_tower matrix is building...")
    # 初始化A矩阵
    tower.initialize_incidence_matrix()

    df_A = pd.DataFrame(tower.incidence_matrix, index=tower.wires_name, columns=tower.wires.get_all_nodes())
 #   print(df_A)
    print("A_tower matrix is built successfully")
    print("------------------------------------------------")

# R
def build_resistance_matrix(tower, Rin, Rx, tube_length):
    # R矩阵
    print("------------------------------------------------")
    print("R_tower matrix is building...")

    tower.initialize_resistance_matrix()

    if tower.tubeWire != None:

        tower.expand_resistance_matrix()

        tower.update_resistance_matrix_by_tubeWires(Rin, Rx, tube_length)

    # df_R = pd.DataFrame(tower.resistance_matrix, index=tower.wires_name, columns=tower.wires_name)
    print("R_tower matrix is built successfully")
    print("------------------------------------------------")

# L
def build_inductance_matrix(tower, L, Lin, Lx, tube_length, sheath_inductance_matrix):
    # L矩阵
    print("------------------------------------------------")
    print("L_tower matrix is building...")

    tower.initialize_inductance_matrix()

    tower.add_inductance_matrix(L)

    if tower.tubeWire != None:

        tower.expand_inductance_matrix()

        tower.update_inductance_matrix_by_coreWires(sheath_inductance_matrix)

        tower.update_inductance_matrix_by_tubeWires(sheath_inductance_matrix, Lin, Lx, tube_length)
    #构建L矩阵df表，便于后续索引
    #tower.inductance_matrix_df = pd.DataFrame(tower.inductance_matrix, index=tower.wires_name, columns=tower.wires_name)
    #print(tower.inductance_matrix)
    print("L_tower matrix is built successfully")
    print("------------------------------------------------")
    return 0

# P
def build_potential_matrix(tower, P, ground_epr):
    # P矩阵
    print("------------------------------------------------")
    print("P_tower matrix is building...")

    tower.initialize_potential_matrix(P)

    tower.update_potential_matrix_by_ground(ground_epr)

    if tower.tubeWire != None:

        tower.update_potential_matrix_by_tubeWires()

    #构建P矩阵df表，便于后续索引
    # df_R = pd.DataFrame(tower.potential_matrix, index=tower.wires_name, columns=tower.wires_name)
    # print(df_R)
    #print(tower.potential_matrix)
    print("P matrix is built successfully")
    print("------------------------------------------------")


def build_conductance_matrix(tower, P, constants, ground_sig):
    # P矩阵
    print("------------------------------------------------")
    print("G_tower matrix is building...")

    tower.initialize_conductance_matrix()

    tower.update_conductance_matrix_by_ground(P, ground_sig, constants)



    # 构建P矩阵df表，便于后续索引
    # df_R = pd.DataFrame(tower.potential_matrix, index=tower.wires_name, columns=tower.wires_name)
    # print(df_R)
    # print(tower.potential_matrix)
    print("G matrix is built successfully")
    print("------------------------------------------------")

#C
def build_capacitance_matrix(tower, Cin):
    # C矩阵
    print("------------------------------------------------")
    print("C_tower matrix is building...")

    tower.initialize_capacitance_matrix()

    if tower.tubeWire != None:

        tower.update_capacitance_matrix_by_tubeWires(Cin)
    #构建P矩阵df表，便于后续索引
   # tower.capacitance_matrix_df = pd.DataFrame(tower.capacitance_matrix, index=tower.wires.get_all_nodes(), columns=tower.wires.get_all_nodes())
    print("C_tower matrix is built successfully")
    print("------------------------------------------------")


def build_impedance_matrix(tower, Lin, sheath_inductance_matrix, tube_length, varied_frequency, constants):
    # 计算套管和芯线内部的阻抗矩阵
    # Core wires impedance

    Nf = varied_frequency.size

    tower.initialize_impedance_matrix(varied_frequency, constants)

    if tower.tubeWire != None:
        Zcf, Zsf, Zcsf, Zscf = calculate_impedance_matrix(tower.tubeWire, varied_frequency, constants)

        tower.expand_impedance_matrix(Nf)

        tower.update_impedance_matrix_by_tubeWires(Zcf, Zsf, Zcsf, Zscf, Lin, sheath_inductance_matrix, tube_length, varied_frequency)


def calculate_impedance_matrix(tubeWire, frequency, constants):
    # 计算套管和芯线内部的阻抗矩阵
    # Core wires impedance
    Zc = calculate_coreWires_impedance(tubeWire.get_coreWires_radii(), tubeWire.get_coreWires_innerOffset(),
                                       tubeWire.get_coreWires_innerAngle(), tubeWire.get_coreWires_mur(),
                                       tubeWire.get_coreWires_sig(), tubeWire.sheath.mur, tubeWire.sheath.sig,
                                       tubeWire.inner_radius, frequency, constants)

    # Sheath impedance
    Zs = calculate_sheath_impedance(tubeWire.sheath.mur, tubeWire.sheath.sig, tubeWire.inner_radius, tubeWire.sheath.r,
                                    frequency, constants)

    # Multual impedance
    Zcs, Zsc = calculate_multual_impedance(tubeWire.get_coreWires_radii(), tubeWire.sheath.mur, tubeWire.sheath.sig,
                                           tubeWire.inner_radius, tubeWire.sheath.r, frequency, constants)

    # 构成套管和芯线内部的阻抗矩阵，其中实部为电阻、虚部为电感，后续将会按照实部和虚部分别取出
    # Zin = np.block([[Zs, Zsc],
    #                 [Zcs, Zc]])

    return Zc, Zs, Zcs, Zsc


def build_tubeWire_inductance_capacitance(tubeWire, constants):
    Vtube = 1e6

    # Core wires inductance
    Lc = calculate_coreWires_inductance(tubeWire.get_coreWires_radii(), tubeWire.get_coreWires_innerOffset(),
                                        tubeWire.get_coreWires_innerAngle(), tubeWire.inner_radius, constants)

    # Core wires capacitance
    Cc = calculate_coreWires_capacitance(tubeWire.outer_radius, tubeWire.inner_radius, tubeWire.get_coreWires_epr(), Lc,
                                         constants)

    # Sheath inductance
    Ls = calculate_sheath_inductance(Vtube, tubeWire.sheath.r, tubeWire.outer_radius,
                                     constants)

    # Sheath capacitance
    Cs = calculate_sheath_capacitance(Vtube, tubeWire.sheath.epr, Ls, constants)

    return Lc, Cc, Ls, Cs


def prepare_building_parameters(tubeWire, frequency, constants):
    Zc, Zs, Zcs, Zsc = calculate_impedance_matrix(tubeWire, frequency, constants)
    Zin = np.block([[Zs, Zsc],
                    [Zcs, Zc]])

    Lc, Cc, Ls, Cs = build_tubeWire_inductance_capacitance(tubeWire, constants)
    # 构成套管和芯线内部的电阻矩阵
    Rin = np.real(Zin)

    # 构成套管和芯线内部的电感矩阵
    Lin = (np.imag(Zin) / (2 * np.pi * frequency) + block_diag(Ls, Lc))

    # 构建套管和芯线内部的电容矩阵
    Cin = block_diag(Cs, Cc)

    # 计算套管和芯线的电感矩阵
    Lx = block_diag(0, (np.tile(np.imag(Zcs) / (2 * np.pi * frequency), (1, 3)) + np.tile(
        np.imag(Zsc) / (2 * np.pi * frequency), (3, 1))))
    # 计算套管和芯线的电阻矩阵
    Rx = block_diag(0, (np.real(Zsc) + np.real(Zcs)))

    return Rin, Rx, Lin, Lx, Cin


def prepare_sheath_inductance(tower):
    # 获取矩阵中表皮开始的索引和结束的索引
    sheath_start_index = len(tower.wires.air_wires) + len(tower.wires.ground_wires)
    sheath_end_index = len(tower.wires.air_wires) + len(tower.wires.ground_wires) + len(tower.wires.tube_wires)
    # 单独获取表皮的电感矩阵
    sheath_inductance_matrix = np.copy(
        tower.inductance_matrix[sheath_start_index:sheath_end_index, sheath_start_index:sheath_end_index])
    return sheath_inductance_matrix


def build_current_source_matrix(tower, I):
    node_name_list = tower.wires.get_all_nodes()
    tower.current_source_matrix = pd.DataFrame(I, index=node_name_list, columns=[0])


def build_voltage_source_matrix(tower, V):
    # 获取线名称列表
    wire_name_list = tower.wires_name
    tower.voltage_source_matrix = pd.DataFrame(V, index=wire_name_list, columns=[0])


def tower_building(tower, fixed_frequency, ground):
    print("------------------------------------------------")
    print(f"Tower:{tower.name} building...")
    # 0.参数准备
    constants = Constant()
    if tower.tubeWire != None:
        Rin, Rx, Lin, Lx, Cin = prepare_building_parameters(tower.tubeWire, fixed_frequency, constants)
        tube_length = tower.wires.get_tube_lengths()[0]
        sheath_inductance_matrix = prepare_sheath_inductance(tower)
    else:
        Rin, Rx, Lin, Lx, Cin = 0, 0, 0, 0, 0
        tube_length = 0
        sheath_inductance_matrix = 0
    L, P = calculate_wires_inductance_potential_with_ground(tower.wires, ground, constants)

    # 1. 构建A矩阵
    build_incidence_matrix(tower)

    # 2. 构建R矩阵
    build_resistance_matrix(tower, Rin, Rx, tube_length)

    # 3. 构建L矩阵
    build_inductance_matrix(tower, L, Lin, Lx, tube_length, sheath_inductance_matrix)

    # 4. 构建P矩阵, node*node
    build_potential_matrix(tower, P, ground.epr)

    # 5. 构建C矩阵
    build_capacitance_matrix(tower, Cin)

    # 6. 构建G矩阵, node*node
    build_conductance_matrix(tower, P, constants, ground.sig)

    build_current_source_matrix(tower, 0)

    build_voltage_source_matrix(tower, 0)

    # 7. 合并lumps和tower
    tower.combine_parameter_matrix()


    print(f"Tower:{tower.name}  building is completed.")
    print("------------------------------------------------")

def build_variant_frequency_matrix(tower, L, Lin, sheath_inductance_matrix, tube_length, varied_frequency, Nfit, dt, constants):
    if tower.tubeWire is not None:

        build_impedance_matrix(tower, Lin, sheath_inductance_matrix, tube_length, varied_frequency, constants)

        SER = matrix_vector_fitting(tower.impedance_matrix, varied_frequency, Nfit)

        A, B = preparing_parameters(SER, dt)

        tower.resistance_matrix = SER['D'] + tower.A.sum(-1)

        tower.inductance_matrix = SER['E'] + L

    else:
        build_impedance_matrix(tower, Lin, sheath_inductance_matrix, tube_length, varied_frequency, constants)

        Ncon = tower.impedance_matrix.shape[0]

        resistance = np.zeros((Ncon, Ncon))
        inductance = np.zeros((Ncon, Ncon))

        A = np.zeros((Ncon, Ncon, Nfit))
        B = np.zeros((Ncon, 1, Nfit))

        for i in range(Ncon):
            SER = matrix_vector_fitting(tower.impedance_matrix[i, i, :].reshape(1, 1, -1), varied_frequency, Nfit)

            SERA, SERB = preparing_parameters(SER, dt)
            A[i, i, :] = SERA
            B[i, 0, :] = SERB

            resistance[i, i] = (SER['D'] + SERA.sum(-1)).squeeze()
            inductance[i, i] = SER['E'].squeeze()

        tower.resistance_matrix = resistance

        tower.inductance_matrix = inductance + L

    tower.phi = np.zeros((len(tower.wires_name), 1, Nfit))

    tower.A = A

    tower.B = B


def tower_building_variant_frequency(tower, frequency, ground, varied_frequency, Nfit, dt):
    print("------------------------------------------------")
    print(f"Tower:{tower.name} building...")
    # 0.参数准备
    constants = Constant()
    if tower.tubeWire is not None:
        Rin, Rx, Lin, Lx, Cin = prepare_building_parameters(tower.tubeWire, frequency, constants)
        tube_length = tower.wires.get_tube_lengths()[0]
        sheath_inductance_matrix = prepare_sheath_inductance(tower)
    else:
        Rin, Rx, Lin, Lx, Cin = 0, 0, 0, 0, 0
        tube_length = 0
        sheath_inductance_matrix = 0
    L, P = calculate_wires_inductance_potential_with_ground(tower.wires, ground, constants)

    # 1. 构建A矩阵
    build_incidence_matrix(tower)

    # 4. 构建P矩阵, node*node
    build_potential_matrix(tower, P, ground.epr)

    # 5. 构建C矩阵
    build_capacitance_matrix(tower, Cin)

    # 6. 构建G矩阵, node*node
    build_conductance_matrix(tower, P, constants, ground.sig)

    build_variant_frequency_matrix(tower, L, Lin, sheath_inductance_matrix, tube_length, varied_frequency, Nfit, dt,
                                   constants)

    build_current_source_matrix(tower, 0)

    build_voltage_source_matrix(tower, (tower.B * tower.phi).sum(-1))

    # 7. 合并lumps和tower
    tower.combine_parameter_matrix()

    print(f"Tower:{tower.name}  building is completed.")
    print("------------------------------------------------")