import numpy as np
import pandas as pd

from Function.Calculators.Impedance import calculate_coreWires_impedance, calculate_sheath_impedance, calculate_multual_impedance, calculate_ground_impedance
from Function.Calculators.Capacitance import calculate_coreWires_capacitance, calculate_sheath_capacitance
from Function.Calculators.Inductance import calculate_coreWires_inductance, calculate_sheath_inductance
from Model.Contant import Constant


def build_incidence_matrix(cable):
    # A矩阵
    print("------------------------------------------------")
    print("A matrix is building...")
    # 初始化A矩阵
    incidence_martix = np.zeros((len(cable.wires_name), len(cable.nodes_name)))
    wires_num = cable.TubeWire.inner_num+1

    for i in range(len(cable.wires_name)):
        incidence_martix[i, i] = -1
        incidence_martix[i, i+wires_num] = 1

    cable.incidence_matrix = pd.DataFrame(incidence_martix, index=cable.wires_name, columns=cable.nodes_name)

    print(cable.incidence_matrix)
    print("A matrix is built successfully")
    print("------------------------------------------------")

def build_resistance_matrix(cable, Zin, Zcs, Zsc, segment_num, segment_length):
    # R矩阵
    print("------------------------------------------------")
    print("R matrix is building...")
    Rin = np.real(Zin)
    Rcs = np.real(Zcs)
    Rsc = np.real(Zsc)
    Rss = Rin[0, 0]
    Rin[0, 0] = 0
    Npha = cable.TubeWire.inner_num
    Rx = np.zeros((1 + Npha, 1 + Npha))
    Rx[1:, 1:] = np.tile(Rsc, (Npha, 1)) + np.tile(Rcs, (1, Npha))
    R = Rin + Rx + Rss

    resistance_martix = np.zeros((len(cable.wires_name), len(cable.wires_name)))

    for i in range(segment_num):
        resistance_martix[i*(Npha+1):(i+1)*(Npha+1), i*(Npha+1):(i+1)*(Npha+1)] = R*segment_length

    cable.resistance_matrix = pd.DataFrame(resistance_martix, index=cable.wires_name, columns=cable.wires_name, dtype=float)

    print(cable.resistance_matrix)
    print("R matrix is built successfully")
    print("------------------------------------------------")

def build_inductance_matrix(cable, Zin, Zcs, Zsc, Lc, Ls, frequency, segment_num, segment_length):
    # L矩阵
    print("------------------------------------------------")
    print("L matrix is building...")
    Npha = cable.TubeWire.inner_num
    Lcs = np.imag(Zcs) / (2 * np.pi * frequency)
    Lsc = np.imag(Zsc) / (2 * np.pi * frequency)
    Ld = np.imag(Zin) / (2 * np.pi * frequency) + np.block(
        [[Ls, np.zeros((1, Lc.shape[1]))], [np.zeros((Lc.shape[0], 1)), Lc]])
    Lss = Ld[0, 0]
    Ld[0, 0] = 0
    Lx = np.zeros((1 + Npha, 1 + Npha))
    Lx[1:, 1:] = np.tile(Lsc, (Npha, 1)) + np.tile(Lcs, (1, Npha))
    L = Ld + Lx + Lss

    inductance_martix = np.zeros((len(cable.wires_name), len(cable.wires_name)))

    for i in range(segment_num):
        inductance_martix[i * (Npha+1):(i + 1) * (Npha+1), i * (Npha+1):(i + 1) * (Npha+1)] = L * segment_length

    cable.inductance_matrix = pd.DataFrame(inductance_martix, index=cable.wires_name, columns=cable.wires_name, dtype=float)

    print(cable.inductance_matrix)
    print("L matrix is built successfully")
    print("------------------------------------------------")

def build_capacitance_matrix(cable, Lc, Ls, constants, segment_num, segment_length):
    # C矩阵
    print("------------------------------------------------")
    print("C matrix is building...")
    Npha = cable.TubeWire.inner_num
    C0 = calculate_coreWires_capacitance(cable.TubeWire.outer_radius, cable.TubeWire.inner_radius,
                                         cable.TubeWire.get_coreWires_epr(), Lc, constants)
    C0se = calculate_sheath_capacitance(cable.TubeWire.get_coreWires_endNodeZ(), cable.TubeWire.sheath.epr, Ls,
                                        constants)
    C = np.block([[C0se, np.zeros((1, C0.shape[1]))], [np.zeros((C0.shape[0], 1)), C0]])
    for jk in range(Npha):
        C[0, jk + 1] = -sum(C[1:, jk + 1])
        C[jk + 1, 0] = -sum(C[jk + 1, 1:])
    C[0, 0] = C[0, 0] - 0.5 * (sum(C[0, 1:]) + sum(C[1:, 0]))

    cable.Cw.C0 = C * (0.5 * segment_length)  # 不知道有什么用途，杜老师说暂时保留

    capacitance_martix = np.zeros((len(cable.nodes_name), len(cable.nodes_name)))

    for i in range(segment_num + 1):
        capacitance_martix[i * (Npha+1):(i + 1) * (Npha+1), i * (Npha+1):(i + 1) * (Npha+1)] = 0.5 * C * segment_length if i == 0 or i == segment_num else C * segment_length
        # 与外界相连接的部分，需要折半

    cable.capacitance_matrix = pd.DataFrame(capacitance_martix, index=cable.nodes_name, columns=cable.nodes_name, dtype=float)

    print(cable.capacitance_matrix)
    print("C matrix is built successfully")
    print("------------------------------------------------")

def build_conductance_matrix(cable):
    # G矩阵
    print("------------------------------------------------")
    print("G matrix is building...")

    cable.conductance_matrix = pd.DataFrame(0, index=cable.nodes_name, columns=cable.nodes_name, dtype=float)

    print(cable.conductance_matrix)
    print("G matrix is built successfully")
    print("------------------------------------------------")


def build_impedance_matrix(cable, Lc, Ls, constants, varied_frequency):
    # Z矩阵
    print("------------------------------------------------")
    print("Z matrix is building...")
    Nf = varied_frequency.size
    Npha = cable.TubeWire.inner_num

    Zgf = calculate_ground_impedance(cable.ground.mur, cable.ground.epr, cable.ground.sig,
                                     cable.TubeWire.get_coreWires_endNodeZ(), cable.TubeWire.outer_radius,
                                     np.zeros((Npha, 1)), varied_frequency, constants)
    Zcf = calculate_coreWires_impedance(cable.TubeWire.get_coreWires_radii(),
                                        cable.TubeWire.get_coreWires_innerOffset(),
                                        cable.TubeWire.get_coreWires_innerAngle(), cable.TubeWire.get_coreWires_mur(),
                                        cable.TubeWire.get_coreWires_sig(), cable.TubeWire.sheath.mur,
                                        cable.TubeWire.sheath.sig, cable.TubeWire.inner_radius, varied_frequency,
                                        constants)
    Zsf = calculate_sheath_impedance(cable.TubeWire.sheath.mur, cable.TubeWire.sheath.sig, cable.TubeWire.inner_radius,
                                     cable.TubeWire.sheath.r, varied_frequency, constants)
    Zcsf, Zscf = calculate_multual_impedance(cable.TubeWire.get_coreWires_radii(), cable.TubeWire.sheath.mur,
                                             cable.TubeWire.sheath.sig, cable.TubeWire.inner_radius,
                                             cable.TubeWire.sheath.r, varied_frequency, constants)
    Zssf = Zsf + Zgf

    Z = np.zeros((Npha + 1, Npha + 1, Nf), dtype='complex')
    for jk in range(Nf):
        Zss = Zssf[0, 0, jk] + 1j * 2 * np.pi * varied_frequency[jk] * Ls
        Z1 = np.block([[0, Zscf[:, :, jk]], [Zcsf[:, :, jk], Zcf[:, :, jk]]])
        Z2 = 1j * 2 * np.pi * varied_frequency[jk] * Lc
        Z3 = np.tile(Zscf[:, :, jk], (Npha, 1)) + np.tile(Zcsf[:, :, jk], (1, Npha))
        Z3 = np.block([[0, np.zeros((1, Npha))], [np.zeros((Npha, 1)), Z2 + Z3]])
        Z[:, :, jk] = Z1 + Z3 + Zss

    cable.impedance_martix = Z

    print("Z matrix is built successfully")
    print("------------------------------------------------")


def build_core_sheath_merged_impedance_matrix(tubeWire, frequency, constants):
    # 计算套管和芯线内部的阻抗矩阵,和tower_modeling中的building_impedance_martix()内容相同
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
    Zin = np.block([[Zs, Zsc],
                    [Zcs, Zc]])
    return Zin, Zcs, Zsc


def cable_building(cable, frequency, varied_frequency, segment_num, segment_length):
    print("------------------------------------------------")
    print("Cable building...")
    # 0.参数准备
    constants = Constant()

    cable.get_brans_nodes_list(segment_num)

    Zin, Zcs, Zsc = build_core_sheath_merged_impedance_matrix(cable.TubeWire, frequency, constants)
    Lc = calculate_coreWires_inductance(cable.TubeWire.get_coreWires_radii(),
                                        cable.TubeWire.get_coreWires_innerOffset(),
                                        cable.TubeWire.get_coreWires_innerAngle(),
                                        cable.TubeWire.inner_radius, constants)

    Ls = calculate_sheath_inductance(cable.TubeWire.get_coreWires_endNodeZ(), cable.TubeWire.sheath.r,
                                     cable.TubeWire.outer_radius, constants)

    # 1. 构建A矩阵
    build_incidence_matrix(cable)

    # 2. 构建R矩阵
    build_resistance_matrix(cable, Zin, Zcs, Zsc, segment_num, segment_length)

    # 3. 构建L矩阵
    build_inductance_matrix(cable, Zin, Zcs, Zsc, Lc, Ls, frequency, segment_num, segment_length)

    # 4. 构建C矩阵
    build_capacitance_matrix(cable, Lc, Ls, constants, segment_num, segment_length)

    # 5. 构建G矩阵
    build_conductance_matrix(cable)

    # 6. 构建Z矩阵
    build_impedance_matrix(cable, Lc, Ls, constants, varied_frequency)
    print("Cable building is completed.")
    print("------------------------------------------------")
