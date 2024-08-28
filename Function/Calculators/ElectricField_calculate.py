import numpy as np
from Model.Lightning import Stroke
from Utils.Math import calculate_electric_field_down_r_and_z
import pandas as pd

def ElectricField_calculate(pt_start, pt_end, stroke: Stroke, ep0, vc):
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

    Ez_air, Er_air = calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img =0)
    Ez_img, Er_img = calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, z_channel_img, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img =1)

    # 合成air和img的电场
    Ez_T = stroke.dh * (Ez_air + Ez_img)
    Er_T = stroke.dh * (Er_air + Er_img)

    return Ez_T, Er_T

if __name__ == "__main__":
    stroke = Stroke('Heidler', duration=0.1, is_calculated=True, hit_pos=[500, 50, 0], parameter_set=None, parameters=None)
    pt_start = pd.read_excel('pt_start.xlsx', header=None)
    pt_start = pt_start.to_numpy()
    pt_end = pd.read_excel('pt_end.xlsx', header=None)
    pt_end = pt_end.to_numpy()
    i_sr = pd.read_excel('i_sr.xlsx', header=None)
    i_sr = i_sr.to_numpy()
    stroke.current_waveform = i_sr
    ep0 = 8.85e-12
    vc = 3e8
    Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, stroke, ep0, vc)

