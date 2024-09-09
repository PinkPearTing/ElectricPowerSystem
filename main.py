import pandas as pd

from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_8"
    varied_frequency = np.arange(0, 37, 9)
    strategy = Strategy.NoLinear()
    network = Network()
    network.run(file_name, strategy)
    print(network.solution)
    Y13_Tower_1 = [abs(i)for i in network.solution.loc["Y13_Tower_1"].tolist()]
    Y02_Tower_1 = [abs(i) for i in network.solution.loc["Y02_Tower_1"].tolist()]

    X05 = network.solution.loc["X05"].tolist()
    X08 = network.solution.loc["X08"].tolist()
    min = [abs(a - b) for a, b in zip(X05, X08)]

    lightning_source = network.sources.loc["X05"].tolist()[:network.Nt]
    # 生成数据
    x = np.arange(0, network.Nt, 1)  # 横坐标数据为从0到10之间，步长为0.1的等差数组
    y = Y13_Tower_1  #

    # 生成图形
    plt.plot(x, lightning_source)
    plt.title('lightning_source')
    plt.show()
    plt.plot(x, y)
    plt.title('ins_i')
    plt.show()
    plt.plot(x, Y02_Tower_1)
    plt.title('top_i')
    plt.show()
    plt.plot(x, min)
    plt.title('ins_v')
    plt.show()

    # 显示图形
    # 2. 接收到所需测试的类型
    #strategy1 = Strategy.baseStrategy()
   # network.update_H()

# y13_tower1
#x05,x08
    from scipy.io import loadmat
    import pandas as pd
    file_path = "C:\\Users\\demo\\Desktop\\PolyU\\电路参数生成矩阵\\VTS\\03Caculate for Large System\\Tower_V9c_yeung\\Tower_V9c_ding2\\集中参数接地\\"
    bran_index_t1 = ['Y0' + str(i+1)+'_Tower_1' for i in range(8)]
    bran_index_t1.extend(['Y13_Tower_1', 'Y14_Tower_1', 'Y15_Tower_1', 'Y09_Tower_1'])
    bran_index_t2 = ['Y0' + str(i+1)+'_Tower_2' for i in range(8)]
    bran_index_t2.extend(['Y13_Tower_2', 'Y14_Tower_2', 'Y15_Tower_2', 'Y09_Tower_2'])
    bran_index_t1.extend(bran_index_t2)
    # bran_index_t1.extend(['Y1001S_splited_1', 'Y3001A_splited_1', 'Y3001B_splited_1', 'Y3001C_splited_1'])
    bran_index_t1.extend(['Y1001S', 'Y3001A', 'Y3001B', 'Y3001C'])
    bran_index_t1.extend(['Y_T12O_S', 'Y_T22O_S', 'Y_T12O_A', 'Y_T22O_A', 'Y_T12O_B', 'Y_T22O_B', 'Y_T12O_C', 'Y_T22O_C'])
    bran_index = bran_index_t1

    node_list_t1 = [1,9,2,5,3,6,4,7,8,10,11,12]
    node_list_t2 = [i + 16 for i in node_list_t1]
    node_index_t1 = ["X{:0>2d}".format(i) for i in node_list_t1]
    node_index_t2 = ["X{:0>2d}".format(i) for i in node_list_t2]
    node_index_t1.extend(node_index_t2)
    node_index_t1.extend(['X01_OHL', 'X02_OHL', 'X03_OHL', 'X04_OHL', 'X17_OHL', 'X18_OHL', 'X19_OHL', 'X20_OHL'])
    node_index = node_index_t1

    A_mat = pd.DataFrame(loadmat(file_path+'A.mat')['A_total'], index=bran_index, columns=node_index)
    R_mat = pd.DataFrame(loadmat(file_path+'R.mat')['R_total'], index=bran_index, columns=bran_index)
    L_mat = pd.DataFrame(loadmat(file_path+'L.mat')['L_total'], index=bran_index, columns=bran_index)
    C_mat = pd.DataFrame(loadmat(file_path+'C.mat')['C_total'], index=node_index, columns=node_index)
    I_out_mat = pd.DataFrame(loadmat(file_path + 'I_out.mat')['Ibt_total2'], index=bran_index)
    V_out_mat = pd.DataFrame(loadmat(file_path + 'V_out.mat')['Vnt_total2'], index=node_index)
    Is_mat = pd.DataFrame(loadmat(file_path + 'Is.mat')['SR_Is'], index=node_index)
    Vs_mat = pd.DataFrame(loadmat(file_path + 'Vs.mat')['SR_Vs'], index=bran_index, dtype=float)
    L_index = ['L_'+ i for i in bran_index+node_index]
    R_index = ['R_'+ i for i in node_index+bran_index]
    LEFT_mat = pd.DataFrame(loadmat(file_path + 'LEFT.mat')['LEFTs'], index=L_index, columns=R_index)
    LEFT_mat.sort_index(axis=0, inplace=True)
    LEFT_mat.sort_index(axis=1, inplace=True)

    bran, node = A_mat.shape
    V_net = network.solution.iloc[:node, :-1]
    I_net = network.solution.iloc[node:, :-1]
    Vs_net = network.sources.iloc[:bran, :2000]
    Is_net = network.sources.iloc[bran:, :2000]

    A_mat.sort_index(axis=0, inplace=True)
    A_mat.sort_index(axis=1, inplace=True)
    R_mat.sort_index(axis=0, inplace=True)
    R_mat.sort_index(axis=1, inplace=True)
    L_mat.sort_index(axis=0, inplace=True)
    L_mat.sort_index(axis=1, inplace=True)
    C_mat.sort_index(axis=0, inplace=True)
    C_mat.sort_index(axis=1, inplace=True)
    I_out_mat.sort_index(axis=0, inplace=True)
    I_out_mat.sort_index(axis=1, inplace=True)
    V_out_mat.sort_index(axis=0, inplace=True)
    V_out_mat.sort_index(axis=1, inplace=True)
    Is_mat.sort_index(axis=0, inplace=True)
    Is_mat.sort_index(axis=1, inplace=True)
    Vs_mat.sort_index(axis=0, inplace=True)
    Vs_mat.sort_index(axis=1, inplace=True)



    A_mat.equals(network.incidence_matrix_A)
    diff_A = A_mat.compare(network.incidence_matrix_A)
    diff_R = R_mat.compare(network.resistance_matrix)
    diff_L = L_mat.compare(network.inductance_matrix)
    diff_C = C_mat.compare(network.capacitance_matrix)
    diff_I_out = I_out_mat.compare(I_net)
    diff_V_out = V_out_mat.compare(V_net)
    diff_Is = Is_mat.compare(Is_net)
    diff_Vs = Vs_mat.compare(Vs_net)
    # diff_LEFT = LEFT_mat.compare(LEFT_py)


    A_mat.index.difference(network.incidence_matrix_A.index)