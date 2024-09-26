from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
class Strategy(ABC):

    def __init__(self):
        self.capacitance_matrix = None

    @abstractmethod
    def apply(self,netwotk,dt,Nt):
        C = np.array(netwotk.capacitance_matrix)  # 点点
        G = np.array(netwotk.conductance_matrix)
        L = np.array(netwotk.inductance_matrix)  # 线线
        R = np.array(netwotk.resistance_matrix)
        ima = np.array(netwotk.incidence_matrix_A)  # 线点
        imb = np.array(netwotk.incidence_matrix_B.T)  # 点线

class Linear(Strategy):
    def apply(self,network,dt):
        print("linear calculation is used")
        C = np.array(network.capacitance_matrix)  # 点点
        G = np.array(network.conductance_matrix)
        L = np.array(network.inductance_matrix)  # 线线
        R = np.array(network.resistance_matrix)
        ima = np.array(network.incidence_matrix_A)  # 线点
        imb = np.array(network.incidence_matrix_B.T)  # 点线
        source = np.array(network.sources)
        nodes = len(network.capacitance_matrix.columns.tolist())
        branches = len(network.inductance_matrix.columns.tolist())
        time_length = len(network.sources.columns.tolist())

        out = np.zeros((branches + nodes, time_length))
        branches, nodes = ima.shape
        for i in range(time_length - 1):
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            Isource = source[:, i + 1].reshape((-1, 1))
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])
            # temp_result = inv_LEFT.dot(RIGHT)
            temp_result = inv_LEFT.dot(Isource + RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
        network.solution = pd.DataFrame(out,
                                        index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

class NonLinear(Strategy):
    def apply(self,network,dt):
        print("Nonlinear calculation is used")
        branches, nodes = network.incidence_matrix_A.shape
        source = np.array(network.sources)
        time_length = len(network.sources.columns.tolist())
        out = np.zeros((branches + nodes, time_length))
        # source = np.array(sources)
        for i in range(time_length - 1):
            C = network.capacitance_matrix.to_numpy()  # 点点
            G = network.conductance_matrix.to_numpy()
            L = network.inductance_matrix.to_numpy()  # 线线
            R = network.resistance_matrix.to_numpy()
            ima = network.incidence_matrix_A.to_numpy()  # 线点
            imb = network.incidence_matrix_B.T.to_numpy()  # 点线
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            Isource = source[:, i + 1].reshape((-1, 1))
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])

            temp_result = inv_LEFT.dot(Isource + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
            temp_result = pd.DataFrame(temp_result,
                                       index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

            t = dt * (i + 1)
            network.update_H(temp_result, t)

        network.solution = pd.DataFrame(out,
                                        index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())


class variant_frequency(Strategy):
    def apply(self, network, dt):
        print("Nonlinear calculation is used")
        branches, nodes = network.incidence_matrix_A.shape
        time_length = len(network.sources.columns.tolist())
        out = np.zeros((branches + nodes, time_length))
        C = network.capacitance_matrix.to_numpy()  # 点点
        G = network.conductance_matrix.to_numpy()
        L = network.inductance_matrix.to_numpy()  # 线线
        R = network.resistance_matrix.to_numpy()
        ima = network.incidence_matrix_A.to_numpy()  # 线点
        imb = network.incidence_matrix_B.T.to_numpy()  # 点线
        # source = np.array(sources)
        for i in range(time_length - 1):
            source = np.array(network.sources)
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            Isource = source[:, i + 1].reshape((-1, 1))
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])

            temp_result = inv_LEFT.dot(Isource + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
            temp_result = pd.DataFrame(temp_result,
                                       index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

            network.update_source_variant_frequency(temp_result, i+2)

        network.solution = pd.DataFrame(out,
                                        index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())


class Change_DE_max(Strategy):
    def apply(self,network,dt):
        network.reverse_H()
        for lump in network.switch_disruptive_effect_models.parameters:
            lump['DE_max'] = 100
        network.calculate(dt)


class Change_Source(Strategy):
    def apply(self,network,dt):
        network.s

class Measurement(Strategy):
    def apply(self,measurement,solution,dt):
        results = {}
        for key, value in measurement.items():
            data_type = value[0]  # 0:'branch',1: 'normal lump'
            measurement_type = value[1]  # 1:'current',2:'voltage'
            # 处理支路名或节点名
            if data_type == 0:
                branch_name = key
                current = solution.loc[branch_name].tolist() if branch_name in solution.index else None
                node1 = solution.loc[value[3]].tolist() if value[3] in solution.index else None
                node2 = solution.loc[value[4]].tolist() if value[4] in solution.index else None
                voltage = [abs(a - b) for a, b in zip(node1, node2)] if node1 and node2 else None
                p = [a * b for a, b in zip(current, voltage)] if current and voltage else None
                E = sum([i * dt for i in p]) if p else None
                if measurement_type == 1:
                    results[(key,value[3],value[4])] = {"current":current}
                elif measurement_type == 2:
                    results[(key,value[3],value[4])] = {"voltage":voltage}
                elif measurement_type == 3:
                    results[(key,value[3],value[4])] = {"P":p}
                elif measurement_type == 4:
                    results[(key,value[3],value[4])] = {"E":E}
                elif measurement_type == 11:
                    results[(key,value[3],value[4])] = {"current":current,
                                                        "voltage":voltage,
                                                        "E":E,"P":p}
            elif data_type == 1:
                lump_name = key
                # 处理支路名可能是列表的情况
                branches = value[3]
                dict_result = {}
                for bran,n1,n2 in zip(value[2],value[3],value[4]):
                    current = solution.loc[bran].tolist() if bran in solution.index else None
                    node1 = solution.loc[n1].tolist() if n1 in solution.index else None
                    node2 = solution.loc[n2].tolist() if n2 in solution.index else None
                    voltage = [abs(a - b) for a, b in zip(node1, node2)] if (node1 and node2)  else None
                    p = [a * b for a, b in zip(current, voltage)] if (current and voltage)  else None
                    E = sum([i*dt for i in p ]) if p else None

                    if measurement_type == 1:
                        dict_result[bran] = {"current":current}
                    elif measurement_type == 2:
                        dict_result[(n1,n2)] = {"voltage":voltage}
                    elif measurement_type == 3:
                        dict_result[(bran,n1,n2)] = {"P": p}
                    elif measurement_type == 4:
                        dict_result[(bran,n1,n2)] ={"current":current,"voltage":voltage,"P":p,"E":E}
                    elif measurement_type == 11:
                        dict_result[(bran,n1,n2)] ={"E":E}

                results[lump_name] = dict_result
        return results

class Monteclarlo(Strategy):
    def apply(self, network):
        print("montecarlo")

