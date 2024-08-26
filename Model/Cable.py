import numpy as np

from Ground import Ground
from Wires import Wire, TubeWire

class Cable:
    def __init__(self, Info, TubeWire: TubeWire, ground: Ground):
        """
        初始化管状线段对象。(同时满足cable中线段的定义)

        inner_radius (float): 不加套管厚度的内部外径
        outer_radius (float): 添加了套管厚度的整体外径
        inner_num (int): 内部芯线的数量
        """
        self.info = Info
        self.ground = ground
        self.TubeWire = TubeWire
        self.wires_name = []
        self.nodes_name = []

        self.R = None
        self.L = None
        self.C = None
        self.Cw = CW(0)
        self.Z = None

    def get_brans_nodes_list(self, segment_num):
        """
        【函数功能】 获取切分后支路列表与节点列表
        【入参】
        segment_num(int):一条线切分后的线段数
        segment_length(float):线长

        【出参】
        brans_name(list,wires_num*segment_num):支路名称列表
        nodes_name(list,wires_num*segment_num+1):节点名称列表
        """
        brans_name = self.TubeWire.get_all_wires()
        start_nodes_name = self.TubeWire.get_all_start_nodes()
        end_nodes_name = self.TubeWire.get_all_end_nodes()
        if segment_num == 1:
            self.wires_name = brans_name
            self.nodes_name = start_nodes_name
            self.nodes_name.extend(end_nodes_name)
        else:
            for i in range(segment_num):
                for j in range(len(brans_name)):
                    self.wires_name.append(f"{brans_name[j]}_Splited_{i+1}")

                self.nodes_name.extend(start_nodes_name)
                for j in range(len(start_nodes_name)):
                    start_nodes_name[j] = f"{brans_name[j]}_MiddleNode_{i + 1}"

            # 最后一个分段，则将终止节点加入列表
            self.nodes_name.extend(end_nodes_name)


class CW:
    def __init__(self, C0):
        self.C0 = C0
