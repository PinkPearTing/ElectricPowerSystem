import numpy as np
from Utils.Math import calculate_H_magnetic_field_down_r
from Model.Lightning import Stroke
import pandas as pd

def H_MagneticField_calculate(pt_start, pt_end, stroke, ep0, vc):
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
    t_sr = np.arange(1, stroke.Nt + 1) * stroke.dt * 1e6
    t_sr = t_sr.reshape(1, -1)

    # 雷电通道每段的中点z坐标和镜像通道的z坐标
    z_channel = (np.arange(1, stroke.N_channel_segment + 1) - 0.5) * stroke.dh
    z_channel_img = -z_channel

    # 计算电流的积分和微分
    i_sr_int = np.cumsum(i_sr) * stroke.dt
    i_sr_int = i_sr_int.reshape(1, -1)
    i_sr_div = np.zeros_like(i_sr)
    i_sr_div[0, 1:] = (np.diff(i_sr) / stroke.dt).reshape(1, -1)
    i_sr_div[0, 0] = i_sr[0, 0] / stroke.dt

    Er_air = calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, 0)
    Er_img = calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, z_channel_img, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, 1)

    # 合成air和img的电场
    Er_T = stroke.dh * (Er_air + Er_img)

    H_all_2 = ep0 * Er_T
    H_p = H_all_2.T

    return H_p


if __name__ == "__main__":
    stroke = Stroke('Heidler', duration=0.1, is_calculated=True, hit_pos=[500, 50, 0], parameter_set=None, parameters=None)
    pt_start = pd.read_excel('pt_start.xlsx', header=None)
    pt_start = pt_start.to_numpy()
    pt_end = pd.read_excel('pt_end.xlsx', header=None)
    pt_end = pt_end.to_numpy()
    i_sr = pd.read_excel('i_sr.xlsx', header=None)
    i_sr = i_sr.to_numpy()
    stroke.current_waveform = i_sr
    stroke.vcof = 1 / 3.0
    ep0 = 8.85e-12
    vc = 3e8
    H_p = H_MagneticField_calculate(pt_start, pt_end, stroke, ep0, vc)