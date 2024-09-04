import numpy as np
from Utils.Math import calculate_distances
import sys
sys.path.append(r'D:\Documents\a实验室\过电压计算程序\ElectricModelBuilding_python\ElectricPowerSystem\Vector_Fitting')
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from Model.Contant import Constant
from Vector_Fitting.Calculators.vecfit_kernel_z import vecfit_kernel_Z_Ding
from Model.Lightning import Lightning, Stroke, Channel
from Utils.Math import calculate_H_magnetic_field_down_r
from Utils.Math import calculate_electric_field_down_r_and_z
import pandas as pd
import math

def distance(node1, node2):
    return math.sqrt((node1.x - node2[0]) ** 2 +
                     (node1.y - node2[1]) ** 2 +
                     (node1.z - node2[2]) ** 2)

def LightningCurrrent_calculate(p1, p2, position, network, node_index, lightning, stroke_sequence):
    """
    【功能】
    计算直击雷的电流源矩阵
    【输入】
    p1 (np.array, (n × 3)):
    :param p2:
    :param position:
    :param network:
    :param node_index:
    :param lightning:
    :param stroke_sequence:
    :return:
    """
    area = p1.split("_")[0]
    # 1. find the wire that is hit
    selected_wire = None
    if area == "tower":

        selected_tower = [tower for tower in network.towers if tower.info.name == p1]
        selected_wire = [wire for wire in list(selected_tower[0].wires.get_all_wires().values()) if wire.name.split("_")[0] == p2]
    elif area == "OHL":
        selected_ohl = [ohl for ohl in network.ohls if ohl.name == p1]
        selected_wire = [wire for wire in list(selected_ohl[0].wires.get_all_wires().values()) if wire.name.split("_")[0] == p2]
    elif area == "cable":
        selected_cable = [cable for cable in network.cables if cable.name == p1]
        selected_wire = [wire for wire in list(selected_cable[0].wires.get_all_wires().values()) if wire.name.split("_")[0] == p2]
    # 2. find the closest node among nodes in the hit wire.
    nodes = set()
    for wire in selected_wire:
        nodes.add(wire.start_node)
        nodes.add(wire.end_node)

    closest_point = None
    min_distance = float('inf')

    for node in nodes:
        dist = distance(node, position)
        if dist < min_distance:
            min_distance = dist
            closest_point = node

    if lightning.type == 'Direct':
        # 初始化一个 DataFrame，行索引为 Nodes，列数为 Nt
        I_out = pd.DataFrame(0, index=node_index, columns=range(lightning.strokes[stroke_sequence].Nt), dtype=np.float64)
        I_out.loc[closest_point.name] = lightning.strokes[stroke_sequence].current_waveform
        return I_out
    elif lightning.type == 'Indirect':
        I_out = pd.DataFrame(0, index=node_index, columns=range(lightning.strokes[stroke_sequence].Nt), dtype=np.float64)
        return I_out


def InducedVoltage_calculate(pt_start, pt_end, branch_list, lightning: Lightning, stroke_sequence, constants: Constant):
    """
    【功能】：
    计算每个导体段，在每个时刻的感应电动势
    【输入】
    conduct_object:  Tower, OHL or Cable对象
    stroke: 雷电对象
    constants: 常数对象
    【输出】
    U_out (len(pt_start), lightning.strokes[stroke_sequence].Nt): 电压矩阵
    """
    if lightning.type == 'Indirect':
        Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, lightning.strokes[stroke_sequence], lightning.channel, constants.ep0, constants.vc)  # 计算电场
        H_p = H_MagneticField_calculate(pt_start, pt_end, lightning.strokes[stroke_sequence], lightning.channel, constants.ep0, constants.vc)  # 计算磁场

        # 计算有损地面的电场
        Er_lossy = ElectricField_above_lossy(-H_p, Er_T, constants, constants.sigma)
        Ez_lossy = Ez_T

        # 利用公式U = E * L计算感应电动势
        a00 = pt_start.shape[0]  # 导体段个数

        Rx = (pt_start[:, 0] + pt_end[:, 0]) / 2 - lightning.channel.hit_pos[0]
        Ry = (pt_start[:, 1] + pt_end[:, 1]) / 2 - lightning.channel.hit_pos[1]
        Rxy = np.sqrt(Rx**2 + Ry**2)

        Uout = np.zeros((lightning.strokes[stroke_sequence].Nt, a00))  # 初始化矩阵

        for ik in range(a00):
            x1, y1, z1 = pt_start[ik]
            x2, y2, z2 = pt_end[ik]

            if Rxy[ik] == 0:
                Uout[:, ik] = Ez_lossy[:, ik] * (z1 - z2)
            else:
                cosx = Rx[ik] / Rxy[ik]
                cosy = Ry[ik] / Rxy[ik]
                Uout[:, ik] = (Er_lossy[:, ik] * cosx * (x1 - x2) +
                               Er_lossy[:, ik] * cosy * (y1 - y2) +
                               Ez_lossy[:, ik] * (z1 - z2))

        Uout = Uout.T
        return pd.DataFrame(Uout, index=branch_list)
    elif lightning.type == 'Direct':
        Uout = pd.DataFrame(0, index=branch_list, columns=range(lightning.strokes[stroke_sequence].Nt), dtype=np.float64)
        return Uout

def ElectricField_calculate(pt_start, pt_end, stroke: Stroke, channel, ep0, vc):
    """
    功能：计雷击影响下，不同时刻每个导体段上r方向和z方向的总电场

    参数说明：
    pt_start (np.array, (n, 3)): 导体段的起点坐标，每行代表坐标(x, y, z)
    pt_end (np.array, (n, 3)): 导体段的终点坐标，每行代表坐标(x, y, z)
    stroke (Stroke对象)
    ep0 (float): 常数，真空介电常数
    vc(float): 常数，光速

    返回：
    Ez_T (np.array, (stroke.Nt,  n): n为导体段的个数
    Er_T (np.array, (stroke.Nt,  n): n为导体段的个数
    """

    # 电流时间序列
    i_sr = stroke.current_waveform
    i_sr = i_sr.reshape(1, -1)

    # 时刻的序列
    # t_sr = np.arange(1, stroke.Nt + 1) * stroke.dt * 1e6
    t_sr = stroke.t_us
    t_sr = t_sr.reshape(1, -1)

    # 雷电通道每段的中点z坐标和镜像通道的z坐标
    z_channel = (np.arange(1, channel.N_channel_segment + 1) - 0.5) * channel.dh
    z_channel_img = -z_channel

    # 计算电流的积分和微分
    i_sr_int = np.cumsum(i_sr) * stroke.dt
    i_sr_int = i_sr_int.reshape(1, -1)
    i_sr_div = np.zeros_like(i_sr)
    i_sr_div[0, 1:] = (np.diff(i_sr) / stroke.dt).reshape(1, -1)
    i_sr_div[0, 0] = i_sr[0, 0] / stroke.dt

    Ez_air, Er_air = calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img =0)
    Ez_img, Er_img = calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, channel, z_channel_img, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img =1)

    # 合成air和img的电场
    Ez_T = channel.dh * (Ez_air + Ez_img)
    Er_T = channel.dh * (Er_air + Er_img)

    return Ez_T, Er_T


def H_MagneticField_calculate(pt_start, pt_end, stroke, channel, ep0, vc):
    # # 常数初始化
    # ep0 = constants.ep0, vc = constants.vc
    # # 时间步长
    # dt = channel.dt
    # # 时间步数
    # Nt = channel.Nt
    # # 通道每段的长度
    # dh = channel.dh
    # # 通道段个数
    # Ns_ch = channel.Nc
    # # 模型选择
    # model_type = channel.model
    # # 与模型相关的参数
    # H = channel.H
    # lamda = channel.lamda
    # vcof = 1.1 / (1 / channel.vcof)

    # 电流时间序列
    i_sr = stroke.current_waveform
    i_sr = i_sr.reshape(1, -1)

    # 时刻的序列
    t_sr = stroke.t_us
    t_sr = t_sr.reshape(1, -1)

    # 雷电通道每段的中点z坐标和镜像通道的z坐标
    z_channel = (np.arange(1, channel.N_channel_segment + 1) - 0.5) * channel.dh
    z_channel_img = -z_channel

    # 计算电流的积分和微分
    i_sr_int = np.cumsum(i_sr) * stroke.dt
    i_sr_int = i_sr_int.reshape(1, -1)
    i_sr_div = np.zeros_like(i_sr)
    i_sr_div[0, 1:] = (np.diff(i_sr) / stroke.dt).reshape(1, -1)
    i_sr_div[0, 0] = i_sr[0, 0] / stroke.dt

    Er_air = calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, 0)
    Er_img = calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, channel, z_channel_img, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, 1)

    # 合成air和img的电场
    Er_T = channel.dh * (Er_air + Er_img)

    H_all_2 = ep0 * Er_T
    H_p = H_all_2.T

    return H_p


def ElectricField_above_lossy(HR0, ER,  constants: Constant, sigma0=None):
    erg = constants.epr
    sigma_g = constants.sigma
    if sigma0 is not None:
        sigma_g = sigma0
    dt = constants.dt
    Nt = constants.Nt

    ep0 = constants.ep0
    u0 = constants.mu0
    vc = constants.vc
    Nd = 9
    w = np.array(
        [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
         10000000]).reshape(1, -1)

    H_in = np.zeros((1, 1, w.size), dtype=complex)
    for ii in range(w.size):
        H_in[:, :, ii] = vc * u0 / np.sqrt(erg + sigma_g / (1j * w[:, ii] * ep0))

    # The vecfit_kernel_Z_Ding function must be defined or replaced by an equivalent fitting routine
    R0, L0, Rn, Ln, Zfit = vecfit_kernel_Z_Ding(H_in, w / (2 * np.pi), Nd)

    # R0_1 = R0 - np.sum(Rn, axis=2)
    # L0_1 = L0
    # R_1 = Rn[0, 0, :Nd]
    # L_1 = Ln[0, 0, :Nd]

    a00, Nt = HR0.shape
    t_ob = Nt * dt
    conv_2 = 2
    dt0 = dt / conv_2

    x = np.linspace(dt, t_ob, Nt)
    y = HR0[:, :Nt]
    xi = np.linspace(dt0, t_ob, 2 * Nt)
    interp_func = interp1d(x, y, kind='cubic', axis=1, fill_value="extrapolate")
    # interp_func = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
    H_save2 = interp_func(xi)

    if a00 == 1:
        H_save2 = H_save2.T

    Ntt = H_save2.shape[1]
    H_all_diff = np.zeros_like(H_save2)
    H_all_diff[:, 0] = H_save2[:, 0] / dt0
    H_all_diff[:, 1:Ntt] = np.diff(H_save2, axis=1) / dt0

    ee0 = R0 * H_save2
    eeL = L0 * H_all_diff

    t00 = Ntt
    Rn2 = Rn[0, :Nd]
    Ln2 = Ln[0, :Nd]
    Rn3 = np.tile(Rn2, (t00, 1))
    Ln3 = np.tile(Ln2, (t00, 1))
    tt00 = np.tile(np.arange(1, t00 + 1).reshape(t00, 1), (1, Nd))
    ee = -Rn3 ** 2 / Ln3 * np.exp(-Rn3 / Ln3 * tt00 * dt0)

    ee_T = ee.T
    shape0_after_convolution = H_save2.shape[0] + ee_T[[1], :].shape[0] - 1
    shape1_after_convolution = H_save2.shape[1] + ee_T[[1], :].shape[1] - 1
    ee_conv = np.zeros((shape0_after_convolution, shape1_after_convolution, Nd))  # 预先定义卷积后矩阵大小，因为H_save2和ee矩阵大小已知
    for jj in range(Nd):
        tmp = convolve2d(H_save2, ee[:, [jj]].T, mode='full', boundary='fill')
        ee_conv[:, :, jj] = dt0 * convolve2d(H_save2, ee[:, [jj]].T, mode='full', boundary='fill')

    ee_conv_sum = np.sum(ee_conv, axis=2)
    ee_all = ee0[:, :Ntt:conv_2] + eeL[:, :Ntt:conv_2] + ee_conv_sum[:, :Ntt:conv_2]
    Er_lossy = ER + ee_all.T

    return Er_lossy


if __name__ == "__main__":
    # 定义lightning
    stroke1 = Stroke('Heidler', duration=1.0e-3, is_calculated=True, parameter_set='0.25/100us', parameters=None)
    stroke1.calculate()
    channel = Channel(hit_pos=[500, 50, 0])
    lightning = Lightning(id=1, type='Direct', strokes=[stroke1], channel=channel)
    # 定义常数
    constants = Constant()
    constants.ep0 = 8.85e-12

    Nodes = ['X01', 'X02', 'X03']
    node = 'X02'
    I = LightningCurrent_calculate(Nodes=Nodes, node=node, lightning=lightning, stroke_sequence=0)
    print(I.loc[node, 999])
    pt_start = pd.read_excel('pt_start.xlsx', header=None)
    pt_start = pt_start.to_numpy()
    pt_end = pd.read_excel('pt_end.xlsx', header=None)
    pt_end = pt_end.to_numpy()
    U_out = InducedVoltage_calculate(pt_start, pt_end, lightning, 0, constants)
    print('END')

    # i_sr = pd.read_excel('i_sr.xlsx', header=None)
    # i_sr = i_sr.to_numpy()
    # stroke.current_waveform = i_sr
    # constants = Constant()
    # constants.ep0 = 8.85e-12
    # U_out = InducedVoltage_calculate(pt_start, pt_end, stroke, constants)