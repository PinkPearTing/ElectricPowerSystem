from Ground import Ground
from Wires import Wire
import numpy as np

class OHL:
    def __init__(self, name: str, Cir_No, Phase, phase_num: int, ground: Ground):
        """
        初始化架空线对象。
        name (str): 线的名称
        Cir_No (int): 线圈回路号
        Phase (str): 线圈相线
        phase_num (int): 线圈相数
        ground(Ground类)：大地类
        """
        self.name = name
        self.wires = []
        self.Cir_No = Cir_No
        self.Phase = Phase
        self.phase_num = phase_num
        self.ground = ground
        self.R = np.zeros((phase_num, phase_num))
        self.L = np.zeros((phase_num, phase_num))
        self.C = np.zeros((phase_num, phase_num))
        self.Z = np.zeros((phase_num, phase_num))

    def add_wire(self, wire: Wire):
        """
        向架空线中添加线段。

        Args:
            wire (Wire): 要添加的线段。
        """
        if len(self.wires) >= self.phase_num:
            raise ValueError("TubeWire can only have {} inner wires, but {} is added.".format(self.phase_num,
                                                                                              len(self.wires) + 1))
        self.wires.append(wire)

    def get_radius(self):
        """
        返回架空线集合的半径矩阵。

        返回:
        radius (numpy.narray, n*1): n条架空线的半径矩阵,每行为某一条架空线的半径
        """
        radius = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            radius[i] = wire.r
        return radius

    def get_end_node_y(self):
        """
        返回架空线末端y值矩阵。

        返回:
        end_node_y (numpy.narray, n*1): n条架空线的末端y值
        """
        end_node_y = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            end_node_y[i] = wire.end_node.y
        return end_node_y

    def get_sig(self):
        """
        返回架空线电导率矩阵。

        返回:
        sig (numpy.narray, n*1): n条架空线的电导率
        """
        sig = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            sig[i] = wire.sig
        return sig

    def get_mur(self):
        """
        返回架空线磁导率。

        返回:
        mur (numpy.narray, n*1): n条架空线的磁导率
        """
        mur = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            mur[i] = wire.mur
        return mur

    def get_epr(self):
        """
        返回架空线相对介电常数。

        返回:
        epr (numpy.narray, n*1): n条架空线的相对介电常数
        """
        epr = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            epr[i] = wire.epr
        return epr

    def get_offset(self):
        """
        返回架空线偏置矩阵。

        返回:
        offset (numpy.narray, n*1): n条线的偏置
        """
        offset = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            offset[i] = wire.offset
        return offset

    def get_height(self):
        """
        返回架空线高度矩阵。

        返回:
        height (numpy.narray, n*1): n条线的高度
        """
        height = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            height[i] = wire.height
        return height

    def get_inductance(self):
        """
        返回线电感。

        返回:
        inductance (numpy.narray, n*1): n条架空线的电感
        """
        inductance = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            inductance[i] = wire.L
        return inductance

    def get_resistance(self):
        """
        返回架空线电阻。

        返回:
        resistance (numpy.narray, n*1): n条架空线的电阻
        """
        resistance = np.zeros((len(self.wires), 1))
        for i, wire in enumerate(self.wires):
            resistance[i] = wire.R
        return resistance
