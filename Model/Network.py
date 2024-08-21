import numpy as np
from Driver.modeling.tower_modeling import tower_building

from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire


class Network:
    def __init__(self, **kwargs):
        self.towers = kwargs.get('towers', [])
        self.cables = kwargs.get('cable', [])
        self.ohls = kwargs.get('ohls', [])
        self.sources = kwargs.get('sources', [])
    def get_H(self,f0,max_length):
        tower_building(self.towers[0], f0, max_length)

        print("得到一个合并的大矩阵H（a，b）")
    def get_souces(self):
        print("得到源u")

    def get_x(self):
        print("x=au+bu结果？")

    def update_H(self,h):
        print("更新H矩阵")


