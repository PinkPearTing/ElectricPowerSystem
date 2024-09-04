from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_8"
    strategy = Strategy.baseStrategy()
    network = Network()
    network.run(file_name, strategy)
    print(network.solution)
    Y13_Tower_1 = [abs(i)for i in network.solution.loc["Y13_Tower_1"].tolist()]
    X05 = network.solution.loc["X05"].tolist()
    X08 = network.solution.loc["X08"].tolist()
    min = [abs(a - b) for a, b in zip(X05, X08)]

    # 生成数据
    x = np.arange(0, 10000, 1)  # 横坐标数据为从0到10之间，步长为0.1的等差数组
    y = Y13_Tower_1[:10000]  #

    # 生成图形
    plt.plot(x, y)

    # 显示图形
    plt.show()
    # 2. 接收到所需测试的类型
    #strategy1 = Strategy.baseStrategy()
   # network.update_H()

# y13_tower1
#x05,x08