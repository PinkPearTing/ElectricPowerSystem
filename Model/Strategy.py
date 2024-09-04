from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
class Strategy(ABC):

    def __init__(self):
        self.capacitance_matrix = None

    @abstractmethod
    def apply(netwotk):
        C = np.array(netwotk.capacitance_matrix)  # 点点
        G = np.array(netwotk.conductance_matrix)
        L = np.array(netwotk.inductance_matrix)  # 线线
        R = np.array(netwotk.resistance_matrix)
        ima = np.array(netwotk.incidence_matrix_A)  # 线点
        imb = np.array(netwotk.incidence_matrix_B.T)  # 点线

class LinearStrategy(Strategy):
    def apply(network):
        C = np.array(network.capacitance_matrix)  # 点点
        G = np.array(network.conductance_matrix)
        L = np.array(network.inductance_matrix)  # 线线
        R = np.array(network.resistance_matrix)
        ima = np.array(network.incidence_matrix_A)  # 线点
        imb = np.array(network.incidence_matrix_B.T)  # 点线
        LEFT = np.block([[-ima, -R - L / network.dt], [G + C / network.dt, -imb]])

        print("solution for linear component")
        return LEFT

class baseStrategy(Strategy):
    def apply(self,network):

        C = np.array(network.capacitance_matrix)#点点
        G = np.array(network.conductance_matrix)
        L = np.array(network.inductance_matrix)#线线
        R = np.array(network.resistance_matrix)
        ima = np.array(network.incidence_matrix_A)#线点
        imb = np.array(network.incidence_matrix_B.T)#点线
        source = np.array(network.sources)
        nodes = len(network.capacitance_matrix.columns.tolist())
        branches = len(network.inductance_matrix.columns.tolist())

        out = np.zeros((branches + nodes, network.Nt))
        branches, nodes = ima.shape
        for i in range(network.Nt - 1):
            Vnode = out[:nodes, i].reshape((-1,1))
            Ibran = out[nodes:, i].reshape((-1,1))
            Isource = source[:,i].reshape((-1,1))
            LEFT = np.block([[-ima, -R - L / network.dt], [G + C / network.dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / network.dt).dot(Ibran)], [(C / network.dt).dot(Vnode)]])
            temp_result = inv_LEFT.dot(Isource + RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:,0]
        network.solution = pd.DataFrame(out, index=network.capacitance_matrix.columns.tolist()+network.inductance_matrix.columns.tolist())

class flash_Insulator(Strategy):
    def apply(self,network):
        network.towers[0].lump

class NonLiear(Strategy):
    def apply(self, network):
        """
        【函数功能】非线性电路求解
        【入参】
        ima(numpy.ndarray:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
        imb(numpy.ndarray:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
        R(numpy.ndarray:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
        L(numpy.ndarray:Nbran*Nbran)：电感矩阵（Nbran：支路数）
        G(numpy.ndarray:Nnode*Nnode)：电导矩阵（Nnode：节点数）
        C(numpy.ndarray:Nnode*Nnode)：电容矩阵（Nnode：节点数）
        sources(numpy.ndarray:(Nbran+Nnode)*Nt)：电源矩阵（Nbran：支路数，Nnode：节点数）
        dt(float)：步长
        Nt(int)：计算总次数

        【出参】
        out(numpy.ndarray:(Nbran+Nnode)*Nt)：计算结果矩阵（Nbran：支路数，Nnode：节点数）
        """

        branches, nodes = network.incidence_matrix_A.shape
        out = np.zeros((branches + nodes, network.Nt))
        source = np.array(network.sources)

        # source = np.array(sources)
        for i in range(network.Nt - 1):
            C = network.capacitance_matrix.to_numpy()  # 点点
            G = network.conductance_matrix.to_numpy()
            L = network.inductance_matrix.to_numpy()  # 线线
            R = network.resistance_matrix.to_numpy()
            ima = network.incidence_matrix_A.to_numpy()  # 线点
            imb = network.incidence_matrix_B.T.to_numpy()  # 点线
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            Isource = source[:,i].reshape((-1,1))
            LEFT = np.block([[-ima, -R - L / network.dt], [G + C / network.dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / network.dt).dot(Ibran)], [(C / network.dt).dot(Vnode)]])

            temp_result = inv_LEFT.dot(Isource + RIGHT)
            #temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
            temp_result = pd.DataFrame(temp_result,
                                       index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

            t = network.dt * (i + 1)
            network.update_H(temp_result, t)

        network.solution = pd.DataFrame(out,
                                     index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

