import numpy as np
from scipy.linalg import block_diag
from Model.Node import Node
from Model.Wires import Wire, Wires, CoreWire, TubeWire
from Model.Ground import Ground
from Model.Contant import Constant
from Model.Tower import Tower
from Driver.initialization.initialization import initialize_tower
from Driver.modeling.tower_modeling import tower_building


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


# （1）--------------------------初始化---------------------------
    print("------------------初始化中--------------------")
    file_name = "01_2"

    # 1. tower 初始化
    tower = initialize_tower(file_name,
                             max_length = max_length)


    # 2. Cable 初始化

    # 3. OHL 初始化

    # 4. lump 初始化

    # 5. Source 初始化

    # 6. 构建网络

    print("------------------初始化结束--------------------")


# （2）--------------------------计算矩阵---------------------------
    tower_building(tower, f0, max_length)



# （3）--------------------------更新矩阵---------------------------




# （4）--------------------------measurement---------------------------



