import numpy as np
import pandas as pd

from Function.Calculators.Impedance import calculate_coreWires_impedance, calculate_sheath_impedance, calculate_multual_impedance, calculate_ground_impedance
from Function.Calculators.Capacitance import calculate_coreWires_capacitance, calculate_sheath_capacitance
from Function.Calculators.Inductance import calculate_coreWires_inductance, calculate_sheath_inductance
from Model.Contant import Constant


def build_incidence_matrix(cable):
    # A矩阵
    print("cable------------------------------------------------")
    print("cableA matrix is building...")
    # 初始化A矩阵
    wires_name = cable.wires_name
    nodes_name = cable.nodes_name
    incidence_martix = np.zeros((len(wires_name), len(nodes_name)))
    wires_num = cable.wires.tube_wires[0].inner_num+1

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num):
        incidence_martix[i, i] = -1
        incidence_martix[i, i+wires_num] = 1

    cable.incidence_matrix = pd.DataFrame(incidence_martix, index=wires_name, columns=nodes_name)

    print(cable.incidence_matrix)
    print("cableA matrix is built successfully")
    print("cable------------------------------------------------")

def build_resistance_matrix(cable, Zin, Zcs, Zsc):
    # R矩阵
    print("cable------------------------------------------------")
    print("cableR matrix is building...")

    Rin = np.real(Zin)
    Rcs = np.real(Zcs)
    Rsc = np.real(Zsc)
    Rss = Rin[0, 0]
    Rin[0, 0] = 0
    Npha = cable.wires.tube_wires[0].inner_num
    Rx = np.zeros((1 + Npha, 1 + Npha))
    Rx[1:, 1:] = np.tile(Rsc, (Npha, 1)) + np.tile(Rcs, (1, Npha))
    R = Rin + Rx + Rss

    resistance_martix = np.zeros((len(cable.wires_name), len(cable.wires_name)))

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num):
        length = cable.wires.tube_wires[i].sheath.length()
        resistance_martix[i*(Npha+1):(i+1)*(Npha+1), i*(Npha+1):(i+1)*(Npha+1)] = R * length

    cable.resistance_matrix = pd.DataFrame(resistance_martix, index=cable.wires_name, columns=cable.wires_name, dtype=float)

    print(cable.resistance_matrix)
    print("cableR matrix is built successfully")
    print("cable------------------------------------------------")

def build_inductance_matrix(cable, Zin, Zcs, Zsc, Lc, Ls, frequency):
    # L矩阵
    print("cable------------------------------------------------")
    print("cableL matrix is building...")
    Npha = cable.wires.tube_wires[0].inner_num
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

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num):
        length = cable.wires.tube_wires[i].sheath.length()
        inductance_martix[i * (Npha+1):(i + 1) * (Npha+1), i * (Npha+1):(i + 1) * (Npha+1)] = L * length

    cable.inductance_matrix = pd.DataFrame(inductance_martix, index=cable.wires_name, columns=cable.wires_name, dtype=float)

    print(cable.inductance_matrix)
    print("cableL matrix is built successfully")
    print("cable------------------------------------------------")

def build_capacitance_matrix(cable, Lc, Ls, constants):
    # C矩阵
    print("cable------------------------------------------------")
    print("cableC matrix is building...")
    tube_wire = cable.wires.tube_wires[0]
    Npha = tube_wire.inner_num
    C0 = calculate_coreWires_capacitance(tube_wire.outer_radius, tube_wire.inner_radius,
                                         tube_wire.get_coreWires_epr(), Lc, constants)
    C0se = calculate_sheath_capacitance(tube_wire.get_coreWires_endNodeZ(), tube_wire.sheath.epr, Ls,
                                        constants)
    C = np.block([[C0se, np.zeros((1, C0.shape[1]))], [np.zeros((C0.shape[0], 1)), C0]])
    for jk in range(Npha):
        C[0, jk + 1] = -sum(C[1:, jk + 1])
        C[jk + 1, 0] = -sum(C[jk + 1, 1:])
    C[0, 0] = C[0, 0] - 0.5 * (sum(C[0, 1:]) + sum(C[1:, 0]))

    cable.Cw.C0 = C * (0.5 * cable.wires.tube_wires[0].sheath.length())  # 不知道有什么用途，杜老师说暂时保留

    capacitance_martix = np.zeros((len(cable.nodes_name), len(cable.nodes_name)))

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num):
        length = cable.wires.tube_wires[i].sheath.length()
        capacitance_martix[i * (Npha+1):(i + 1) * (Npha+1), i * (Npha+1):(i + 1) * (Npha+1)] = 0.5 * C * length if i == 0 or i == segment_num else C * length
        # 与外界相连接的部分，需要折半

    cable.capacitance_matrix = pd.DataFrame(capacitance_martix, index=cable.nodes_name, columns=cable.nodes_name, dtype=float)

    print(cable.capacitance_matrix)
    print("cableC matrix is built successfully")
    print("cable------------------------------------------------")

def build_conductance_matrix(cable):
    # G矩阵
    print("cable------------------------------------------------")
    print("cableG matrix is building...")

    cable.conductance_matrix = pd.DataFrame(0, index=cable.nodes_name, columns=cable.nodes_name, dtype=float)

    print(cable.conductance_matrix)
    print("cableG matrix is built successfully")
    print("cable------------------------------------------------")


def build_impedance_matrix(cable, Lc, Ls, constants, varied_frequency):
    # Z矩阵
    print("cable------------------------------------------------")
    print("cableZ matrix is building...")
    Nf = varied_frequency.size
    tube_wire = cable.wires.tube_wires[0]
    Npha = tube_wire.inner_num

    Zgf = calculate_ground_impedance(cable.ground.mur, cable.ground.epr, cable.ground.sig,
                                     tube_wire.get_coreWires_endNodeZ(), tube_wire.outer_radius,
                                     np.zeros((Npha, 1)), varied_frequency, constants)
    Zcf = calculate_coreWires_impedance(tube_wire.get_coreWires_radii(),
                                        tube_wire.get_coreWires_innerOffset(),
                                        tube_wire.get_coreWires_innerAngle(), tube_wire.get_coreWires_mur(),
                                        tube_wire.get_coreWires_sig(), tube_wire.sheath.mur,
                                        tube_wire.sheath.sig, tube_wire.inner_radius, varied_frequency,
                                        constants)
    Zsf = calculate_sheath_impedance(tube_wire.sheath.mur, tube_wire.sheath.sig, tube_wire.inner_radius,
                                     tube_wire.sheath.r, varied_frequency, constants)
    Zcsf, Zscf = calculate_multual_impedance(tube_wire.get_coreWires_radii(), tube_wire.sheath.mur,
                                             tube_wire.sheath.sig, tube_wire.inner_radius,
                                             tube_wire.sheath.r, varied_frequency, constants)
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

    print("cableZ matrix is built successfully")
    print("cable------------------------------------------------")


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


def cable_building(cable, frequency, varied_frequency):
    print("cable------------------------------------------------")
    print("cableCable building...")
    # 0.参数准备
    constants = Constant()
    tube_wire = cable.wires.tube_wires[0]

    Zin, Zcs, Zsc = build_core_sheath_merged_impedance_matrix(tube_wire, frequency, constants)
    Lc = calculate_coreWires_inductance(tube_wire.get_coreWires_radii(),
                                        tube_wire.get_coreWires_innerOffset(),
                                        tube_wire.get_coreWires_innerAngle(),
                                        tube_wire.inner_radius, constants)

    Ls = calculate_sheath_inductance(tube_wire.get_coreWires_endNodeZ(), tube_wire.sheath.r,
                                     tube_wire.outer_radius, constants)

    # 1. 构建A矩阵
    build_incidence_matrix(cable)

    # 2. 构建R矩阵
    build_resistance_matrix(cable, Zin, Zcs, Zsc)

    # 3. 构建L矩阵
    build_inductance_matrix(cable, Zin, Zcs, Zsc, Lc, Ls, frequency)

    # 4. 构建C矩阵
    build_capacitance_matrix(cable, Lc, Ls, constants)

    # 5. 构建G矩阵
    build_conductance_matrix(cable)

    # 6. 构建Z矩阵
    build_impedance_matrix(cable, Lc, Ls, constants, varied_frequency)
    print("cableCable building is completed.")
    print("cable------------------------------------------------")
