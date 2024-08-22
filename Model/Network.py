import numpy as np
import pandas as pd

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

    def combine_parameter_martix(self):
        #按照towers，cables，ohls顺序合并参数矩阵
        #合并sources矩阵
        incidence_matrix = pd.DataFrame()
        resistance_matrix = pd.DataFrame()
        inductance_matrix = pd.DataFrame()
        capacitance_matrix = pd.DataFrame()
        conductance_martix = pd.DataFrame()
        voltage_source_martix = pd.DataFrame()
        current_source_martix = pd.DataFrame()
        for model_list in [self.towers, self.cables, self.ohls]:
            for model in model_list:
                incidence_matrix.add(model.incidence_matrix, fill_value=0).fillna(0)
                resistance_matrix.add(model.resistance_matrix, fill_value=0).fillna(0)
                inductance_matrix.add(model.inductance_matrix, fill_value=0).fillna(0)
                capacitance_matrix.add(model.capacitance_matrix, fill_value=0).fillna(0)
                conductance_martix.add(model.conductance_martix, fill_value=0).fillna(0)
                voltage_source_martix.add(model.voltage_source_martix, fill_value=0).fillna(0)
                current_source_martix.add(model.current_source_martix, fill_value=0).fillna(0)

        return incidence_matrix, resistance_matrix, inductance_matrix, capacitance_matrix, conductance_martix, voltage_source_martix, current_source_martix
