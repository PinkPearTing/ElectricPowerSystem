import numpy as np

from Driver.initialization.initialization import initialize_cable
from Driver.modeling.cable_modeling import cable_building


if __name__ == '__main__':
    # -----------------临时定义的数，后期会改-------------------
    # 变频下的频率矩阵
    frq = np.concatenate([
        np.arange(1, 91, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, 10000, 1000),
        np.arange(10000, 100000, 10000)
    ])
    VF = {'odc': 10,
          'frq': frq}
    # 固频的频率值
    f0 = 2e4
    # 线段的最大长度, 后续会按照这个长度, 对不符合长度规范的线段进行切分
    max_length = 50

    frq_default = np.logspace(0, 9, 37)


# （1）--------------------------初始化---------------------------
    print("------------------初始化中--------------------")
    file_name = "01_2"

    # 1. OHL 初始化
    cable = initialize_cable(file_name, max_length)

    print("------------------初始化结束--------------------")


# （2）--------------------------计算矩阵---------------------------
    segment_num = int(3) #正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
    segment_length = 20 #预设的参数
    cable_building(cable, f0, frq_default, segment_num, segment_length, max_length)


    print(1)


# （3）--------------------------更新矩阵---------------------------




# （4）--------------------------measurement---------------------------


