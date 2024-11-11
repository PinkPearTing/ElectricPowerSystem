import numpy as np
import pandas as pd
import scipy as sp

from Function.Calculators.Impedance import calculate_coreWires_impedance, calculate_sheath_impedance, \
    calculate_multual_impedance, calculate_ground_impedance, calculate_sheath_internal_impedance, \
    calculate_inductance_of_round_wires_inside_sheath, calculate_round_wires_internal_impedance
from Function.Calculators.Capacitance import calculate_coreWires_capacitance, calculate_sheath_capacitance
from Function.Calculators.Inductance import calculate_coreWires_inductance, calculate_sheath_inductance
from Model.Contant import Constant
from Vector_Fitting.Drivers.MatrixFitting import matrix_vector_fitting
from Driver.modeling_varient_freqency.recursive_convolution import preparing_parameters


def build_incidence_matrix(cable):
    # A矩阵
    print("cable------------------------------------------------")
    print("cableA matrix is building...")
    # 初始化A矩阵
    wires_name = cable.wires_name
    nodes_name = cable.nodes_name
    incidence_matrix = np.zeros((len(wires_name), len(nodes_name)))
    wires_num = cable.wires.tube_wires[0].inner_num+1

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num*wires_num):
        incidence_matrix[i, i] = -1
        incidence_matrix[i, i+wires_num] = 1

    cable.incidence_matrix = pd.DataFrame(incidence_matrix, index=wires_name, columns=nodes_name)

    if 'ref' in cable.nodes_name:
        cable.incidence_matrix.drop('ref', axis=1, inplace=True)

    print(cable.incidence_matrix)
    print("cableA matrix is built successfully")
    print("cable------------------------------------------------")

def build_resistance_matrix(cable, R):
    # R矩阵
    print("cable------------------------------------------------")
    print("cableR matrix is building...")
    Npha = cable.wires.tube_wires[0].inner_num

    resistance_matrix = np.zeros((len(cable.wires_name), len(cable.wires_name)))

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num):
        length = cable.wires.tube_wires[i].sheath.length()
        resistance_matrix[i*(Npha+1):(i+1)*(Npha+1), i*(Npha+1):(i+1)*(Npha+1)] = R * length

    cable.resistance_matrix = pd.DataFrame(resistance_matrix, index=cable.wires_name, columns=cable.wires_name, dtype=float)

    print(cable.resistance_matrix)
    print("cableR matrix is built successfully")
    print("cable------------------------------------------------")

def build_inductance_matrix(cable, L):
    # L矩阵
    print("cable------------------------------------------------")
    print("cableL matrix is building...")
    Npha = cable.wires.tube_wires[0].inner_num

    inductance_matrix = np.zeros((len(cable.wires_name), len(cable.wires_name)))

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num):
        length = cable.wires.tube_wires[i].sheath.length()
        inductance_matrix[i * (Npha+1):(i + 1) * (Npha+1), i * (Npha+1):(i + 1) * (Npha+1)] = L * length

    cable.inductance_matrix = pd.DataFrame(inductance_matrix, index=cable.wires_name, columns=cable.wires_name, dtype=float)

    print(cable.inductance_matrix)
    print("cableL matrix is built successfully")
    print("cable------------------------------------------------")

def build_capacitance_matrix(cable, C):
    # C矩阵
    print("cable------------------------------------------------")
    print("cableC matrix is building...")
    tube_wire = cable.wires.tube_wires[0]
    Npha = tube_wire.inner_num
    # C0 = calculate_coreWires_capacitance(tube_wire.outer_radius, tube_wire.inner_radius,
    #                                      tube_wire.get_coreWires_epr(), Lc, constants)
    # C0se = calculate_sheath_capacitance(tube_wire.get_coreWires_endNodeZ(), tube_wire.sheath.epr, Ls,
    #                                     constants)
    # C = np.block([[C0se, np.zeros((1, C0.shape[1]))], [np.zeros((C0.shape[0], 1)), C0]])
    # for jk in range(Npha):
    #     C[0, jk + 1] = -sum(C[1:, jk + 1])
    #     C[jk + 1, 0] = -sum(C[jk + 1, 1:])
    # C[0, 0] = C[0, 0] - 0.5 * (sum(C[0, 1:]) + sum(C[1:, 0]))

    cable.Cw.C0 = C * (0.5 * cable.wires.tube_wires[0].sheath.length())  # 不知道有什么用途，杜老师说暂时保留

    capacitance_matrix = np.zeros((len(cable.nodes_name), len(cable.nodes_name)))

    segment_num = len(cable.wires.tube_wires)

    for i in range(segment_num+1):
        length = cable.wires.tube_wires[0].sheath.length()
        capacitance_matrix[i * (Npha+1):(i + 1) * (Npha+1), i * (Npha+1):(i + 1) * (Npha+1)] = 0.5 * C * length if i == 0 or i == segment_num else C * length
        # 与外界相连接的部分，需要折半

    cable.capacitance_matrix = pd.DataFrame(capacitance_matrix, index=cable.nodes_name, columns=cable.nodes_name, dtype=float)

    if 'ref' in cable.nodes_name:
        cable.capacitance_matrix.drop('ref', axis=0, inplace=True)
        cable.capacitance_matrix.drop('ref', axis=1, inplace=True)

    print(cable.capacitance_matrix)
    print("cableC matrix is built successfully")
    print("cable------------------------------------------------")

def build_conductance_matrix(cable):
    # G矩阵
    print("cable------------------------------------------------")
    print("cableG matrix is building...")

    cable.conductance_matrix = pd.DataFrame(0, index=cable.nodes_name, columns=cable.nodes_name, dtype=float)

    if 'ref' in cable.nodes_name:
        cable.conductance_matrix.drop('ref', axis=0, inplace=True)
        cable.conductance_matrix.drop('ref', axis=1, inplace=True)

    print(cable.conductance_matrix)
    print("cableG matrix is built successfully")
    print("cable------------------------------------------------")


def build_impedance_matrix(cable, constants, varied_frequency, ground):
    # Z矩阵
    print("cable------------------------------------------------")
    print("cableZ matrix is building...")
    Nf = varied_frequency.size
    tube_wire = cable.wires.tube_wires[0]
    Npha = tube_wire.inner_num

    Zgf = calculate_ground_impedance(ground.mur, ground.epr, ground.sig,
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
        # Zss = Zssf[0, 0, jk] + 1j * 2 * np.pi * varied_frequency[jk] * Ls
        Zss = Zssf[0, 0, jk]
        Z1 = np.block([[0, Zscf[:, :, jk]], [Zcsf[:, :, jk], Zcf[:, :, jk]]])
        # Z2 = 1j * 2 * np.pi * varied_frequency[jk] * Lc
        Z3 = np.tile(Zscf[:, :, jk], (Npha, 1)) + np.tile(Zcsf[:, :, jk], (1, Npha))
        # Z3 = np.block([[0, np.zeros((1, Npha))], [np.zeros((Npha, 1)), Z2 + Z3]])
        Z3 = np.block([[0, np.zeros((1, Npha))], [np.zeros((Npha, 1)), Z3]])
        Z[:, :, jk] = Z1 + Z3 + Zss

    cable.impedance_matrix = Z

    print("cableZ matrix is built successfully")
    print("cable------------------------------------------------")


def build_core_sheath_merged_impedance_matrix(tubeWire, frequency, constants):
    # 计算套管和芯线内部的阻抗矩阵,和tower_modeling中的building_impedance_matrix()内容相同
    # Core wires impedance
    # Zc = calculate_coreWires_impedance(tubeWire.get_coreWires_radii(), tubeWire.get_coreWires_innerOffset(),
    #                                    tubeWire.get_coreWires_innerAngle(), tubeWire.get_coreWires_mur(),
    #                                    tubeWire.get_coreWires_sig(), tubeWire.get_coreWires_epr(), tubeWire.sheath.mur, tubeWire.sheath.sig, tubeWire.sheath.epr,
    #                                    tubeWire.inner_radius, frequency, constants)
    omega = 2 * np.pi * frequency
    Nf = frequency.size
    sheath_inner_radius = tubeWire.inner_radius
    core_wires_r = tubeWire.get_coreWires_radii()

    Zcore_diag = calculate_round_wires_internal_impedance(core_wires_r, tubeWire.get_coreWires_mur(), tubeWire.get_coreWires_sig(), tubeWire.get_coreWires_epr(), frequency, constants)

    Lcs = calculate_inductance_of_round_wires_inside_sheath(core_wires_r, tubeWire.get_coreWires_innerOffset(), tubeWire.get_coreWires_innerAngle(), sheath_inner_radius, constants)

    Zsi = calculate_sheath_internal_impedance(tubeWire.sheath.mur, tubeWire.sheath.sig, tubeWire.sheath.epr, sheath_inner_radius, tubeWire.sheath.r, frequency, constants)

    Nc = core_wires_r.shape[0]
    Zc = np.zeros((Nc, Nc, Nf), dtype='complex')
    for ik in range(Nf):
        Zc[:, :, ik] = np.diag(Zcore_diag[:, ik]) + 1j * omega[ik] * Lcs + Zsi[ik]

    # Sheath impedance
    Zs = calculate_sheath_impedance(tubeWire.sheath.mur, tubeWire.sheath.sig, tubeWire.inner_radius, tubeWire.sheath.r, tubeWire.outer_radius,
                                    frequency, constants)

    # Multual impedance
    Zcs, Zsc = calculate_multual_impedance(tubeWire.get_coreWires_radii(), tubeWire.sheath.mur, tubeWire.sheath.sig,
                                           tubeWire.inner_radius, tubeWire.sheath.r, tubeWire.sheath.epr, frequency, constants)

    # 构成套管和芯线内部的阻抗矩阵，其中实部为电阻、虚部为电感，后续将会按照实部和虚部分别取出
    return Zc, Zs, Zcs, Zsc


def build_current_source_matrix(cable, I):
    node_name_list = cable.wires.get_all_nodes()
    cable.current_source_matrix = pd.DataFrame(I, index=node_name_list, columns=[0])
    if 'ref' in cable.nodes_name:
        cable.current_source_matrix.drop('ref', axis=0, inplace=True)


def build_voltage_source_matrix(cable, V):
    # 获取线名称列表
    wire_name_list = cable.wires_name
    cable.voltage_source_matrix = pd.DataFrame(V, index=wire_name_list, columns=[0])

def calculate_cable_resistance(Rin, Rcs, Rsc, Npha):
    Rss = np.copy(Rin[0, 0])
    Rin[0, 0] = 0
    Rx = np.zeros((1 + Npha, 1 + Npha))
    Rx[1:, 1:] = np.tile(Rsc, (Npha, 1)) + np.tile(Rcs, (1, Npha))
    R = Rin + Rx + Rss
    return R

def calculate_cable_inductance(Lin, Lcs, Lsc, Npha):
    # Ld = Lin + sp.linalg.block_diag(Ls, Lc)
    Ld = Lin
    Lss = np.copy(Ld[0, 0])
    Ld[0, 0] = 0
    Lx = np.zeros((1 + Npha, 1 + Npha))
    Lx[1:, 1:] = np.tile(Lsc, (Npha, 1)) + np.tile(Lcs, (1, Npha))
    L = Ld + Lx + Lss
    return L

def prepare_building_parameters(cable, Lc, Ls, ground, frequency, constants):
    tube_wire = cable.wires.tube_wires[0]
    Npha = tube_wire.inner_num

    Zc, Zs, Zcs, Zsc = build_core_sheath_merged_impedance_matrix(tube_wire, frequency, constants)
    Zc = Zc.squeeze(-1)
    Zs = Zs.squeeze(-1)
    Zcs = Zcs.squeeze(-1)
    Zsc = Zsc.squeeze(-1)

    Zg = calculate_ground_impedance(ground.mur, ground.epr, ground.sig,
                                     tube_wire.get_coreWires_endNodeZ(), tube_wire.outer_radius,
                                     np.zeros((Npha, 1)), frequency,
                                     constants)
    Zg = Zg.squeeze(-1)

    Zin = np.block([[Zs + Zg, Zsc], [Zcs, Zc]])

    Rin = np.copy(np.real(Zin))
    Rcs = np.copy(np.real(Zcs))
    Rsc = np.copy(np.real(Zsc))
    R = calculate_cable_resistance(Rin, Rcs, Rsc, Npha)

    Lin = np.copy(np.imag(Zin)) / (2 * np.pi * frequency)
    Lcs = np.copy(np.imag(Zcs)) / (2 * np.pi * frequency)
    Lsc = np.copy(np.imag(Zsc)) / (2 * np.pi * frequency)
    L = calculate_cable_inductance(Lin, Lcs, Lsc, Npha)

    C = calculate_cable_capcitance(tube_wire.get_coreWires_radii(), tube_wire.get_coreWires_innerOffset(),
                                       tube_wire.get_coreWires_innerAngle(),
                                       tube_wire.inner_radius, tube_wire.outer_radius, tube_wire.sheath.r, constants)

    return R, L, C

def calculate_cable_capcitance(core_wires_r, core_wires_offset, core_wires_angle, sheath_inner_radius, sheath_outer_radius, sheath_r, constants):
    mu0, ep0 = constants.mu0, constants.ep0
    Npha = core_wires_r.shape[0]
    tmat = np.tile(core_wires_angle, (1, Npha))
    angle = (tmat - tmat.T) * np.pi / 180
    didk = core_wires_offset @ core_wires_offset.T
    k = 2 * np.pi * ep0

    dj = np.tile(core_wires_offset, (1, Npha))
    P = np.log(dj.T/sheath_inner_radius*np.sqrt((didk**2+sheath_inner_radius**4-2*didk*sheath_inner_radius**2*np.cos(angle))/(didk**2+dj.T**4-2*didk*dj.T**2*np.cos(angle))))/k
    P_diag = np.log(sheath_inner_radius/core_wires_r*(1-(core_wires_offset/sheath_inner_radius)**2))/k
    np.fill_diagonal(P, P_diag)

    Ppip_ins = np.log(sheath_outer_radius/sheath_r)/k
    P = sp.linalg.block_diag([0], P) + Ppip_ins

    return np.linalg.inv(P)


def cable_building(cable, ground, frequency):
    print("cable------------------------------------------------")
    print("cableCable building...")
    # 0.参数准备
    constants = Constant()
    tube_wire = cable.wires.tube_wires[0]

    Lc = calculate_inductance_of_round_wires_inside_sheath(tube_wire.get_coreWires_radii(),
                                                           tube_wire.get_coreWires_innerOffset(),
                                                           tube_wire.get_coreWires_innerAngle(),
                                                           tube_wire.inner_radius, constants)

    Ls = calculate_sheath_inductance(tube_wire.get_coreWires_endNodeZ(), tube_wire.sheath.r,
                                     tube_wire.outer_radius, constants)

    R, L, C = prepare_building_parameters(cable, Lc, Ls, ground, frequency, constants)

    # 1. 构建A矩阵
    build_incidence_matrix(cable)

    # 2. 构建R矩阵
    build_resistance_matrix(cable, R)

    # 3. 构建L矩阵
    build_inductance_matrix(cable, L)

    # 4. 构建C矩阵
    # build_capacitance_matrix(cable, Lc, Ls, constants)
    build_capacitance_matrix(cable, C)

    # 5. 构建G矩阵
    build_conductance_matrix(cable)

    build_current_source_matrix(cable, 0)

    build_voltage_source_matrix(cable, 0)

    print("cableCable building is completed.")
    print("cable------------------------------------------------")


def prepare_variant_frequency_parameters(cable, Lc, Ls, varied_frequency, Nfit, ground, dt, constants):
    tube_wire = cable.wires.tube_wires[0]
    length = cable.wires.tube_wires[0].sheath.length()
    Npha = tube_wire.inner_num
    Nf = varied_frequency.size

    if cable.info.mode_con == 1:
        Zgf = calculate_ground_impedance(ground.mur, ground.epr, ground.sig,
                                         tube_wire.get_coreWires_endNodeZ(), tube_wire.outer_radius,
                                         np.zeros((Npha, 1)), varied_frequency,
                                         constants) if cable.info.mode_gnd == 2 else 0
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

        Zss = Zssf[0, 0, :]
        Z1_up = np.concatenate((np.zeros((1, 1, Nf)), Zscf), axis=1)
        Z1_down = np.concatenate((Zcsf, Zcf), axis=1)
        Z1 = np.concatenate((Z1_up, Z1_down), axis=0)
        Z3 = np.tile(Zscf, (Npha, 1, 1)) + np.tile(Zcsf, (1, Npha, 1))
        Z3_down = np.concatenate((np.zeros((Npha, 1, Nf)), Z3), axis=1)
        Z3 = np.concatenate((np.zeros((1, Npha+1, Nf)), Z3_down), axis=0)
        Z = Z1 + Z3 + Zss

        SER = matrix_vector_fitting(Z, varied_frequency)
        A, B = preparing_parameters(SER, dt)
        resistance = SER['D'] + cable.A.sum(-1)
        inductance = SER['E'] + sp.linalg.block_diag(Ls, Lc)
    elif cable.info.mode_gnd == 2:
        Zgf = calculate_ground_impedance(ground.mur, ground.epr, ground.sig,
                                         tube_wire.get_coreWires_endNodeZ(), tube_wire.outer_radius,
                                         np.zeros((Npha, 1)), varied_frequency, constants)

        SER = matrix_vector_fitting(Zgf, varied_frequency)
        SER['D'] = np.tile(SER['D'], (Npha+1, Npha+1, 1))

        A, B = preparing_parameters(SER, dt)
        A = np.tile(A, (Npha+1, Npha+1, 1))
        R, L = prepare_building_parameters(cable, Lc, Ls, 2e4, constants)
        resistance = SER['D'] + cable.A.sum(-1) + R
        inductance = SER['E'] + L
    else:
        raise Exception('Conductor model and ground model of Cable are not variant frequency, since variant frequency model have used.')

    cable.phi = np.zeros((len(cable.wires_name), 1, Nfit))

    cable.A = A * length

    cable.B = B

    return resistance, inductance

def cable_building_variant_frequency(cable, ground, varied_frequency, dt, Nfit=9):
    print("cable------------------------------------------------")
    print("cableCable building...")
    # 0.参数准备
    constants = Constant()
    tube_wire = cable.wires.tube_wires[0]

    Lc = calculate_inductance_of_round_wires_inside_sheath(tube_wire.get_coreWires_radii(),
                                                           tube_wire.get_coreWires_innerOffset(),
                                                           tube_wire.get_coreWires_innerAngle(),
                                                           tube_wire.inner_radius, constants)

    Ls = calculate_sheath_inductance(tube_wire.get_coreWires_endNodeZ(), tube_wire.sheath.r,
                                     tube_wire.outer_radius, constants)

    R, L = prepare_variant_frequency_parameters(cable, Lc, Ls, varied_frequency, Nfit, ground, dt, constants)

    # 1. 构建A矩阵
    build_incidence_matrix(cable)

    # 4. 构建C矩阵
    build_capacitance_matrix(cable, Lc, Ls, constants)

    # 5. 构建G矩阵
    build_conductance_matrix(cable)

    # 2. 构建R矩阵
    build_resistance_matrix(cable, R)

    # 3. 构建L矩阵
    build_inductance_matrix(cable, L)

    build_current_source_matrix(cable, 0)

    build_voltage_source_matrix(cable, 0)

    print("cableCable building is completed.")
    print("cable------------------------------------------------")
