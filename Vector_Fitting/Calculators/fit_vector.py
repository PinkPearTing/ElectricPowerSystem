import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize  # 仅作为示例，具体需要根据算法细节

# 472行 556 559

def vectfit3(f, s, poles, weight, opts):
    # 初始化输出变量
    # 默认参数值
    defaults = {
        "relax": 1,  # 使用放松的非平凡性约束进行矢量拟合
        "stable": 1,  # 强制稳定极点
        "asymp": 2,  # 仅在拟合中包含D（不包括E）
        "skip_pole": 0,  # 不跳过极点识别
        "skip_res": 0,  # 不跳过残差（C,D,E）的识别
        "cmplx_ss": 1,  # 创建复数状态空间模型
        'spy1': 0,
        'spy2': 1,
        'logx': 1,
        'logy': 1,
        'errplot': 1,
        'phaseplot': 0,
        'legend': 1
    }

    # 如果opts未提供，使用默认值；否则，合并opts和默认值
    if opts is None:
        opts = defaults
    else:
        # 将默认值合并到opts中，如果opts中缺少某个键
        for key, value in defaults.items():
            if key not in opts:
                opts[key] = value

    # 放松矢量拟合使用的容差
    TOLlow = 1e-18
    TOLhigh = 1e18

    # 调整极点位置
    a = len(poles)
    if np.squeeze(s)[0] == 0 and a == 1:
        if poles[0] == 0 and poles[1] != 0:
            poles[0] = -1
        elif poles[1] == 0 and poles[0] != 0:
            poles[1] = -1
        elif poles[0] == 0 and poles[1] == 0:
            poles[0] = -1 + 1j * 10
            poles[1] = -1 - 1j * 10

    # 检查opts字典参数设置的有效性，直接在主体代码中进行
    error_occurred = False  # 用于标记是否发生错误

    if opts['relax'] not in [0, 1]:
        print(f"ERROR in vectfit3: ==> Illegal value for opts.relax: {opts['relax']}")
        error_occurred = True
    if opts['asymp'] not in [1, 2, 3]:
        print(f"ERROR in vectfit3: ==> Illegal value for opts.asymp: {opts['asymp']}")
        error_occurred = True
    if opts['stable'] not in [0, 1]:
        print(f"ERROR in vectfit3: ==> Illegal value for opts.stable: {opts['stable']}")
        error_occurred = True
    if opts['skip_pole'] not in [0, 1]:
        print(f"ERROR in vectfit3: ==> Illegal value for opts.skip_pole: {opts['skip_pole']}")
        error_occurred = True
    if opts['skip_res'] not in [0, 1]:
        print(f"ERROR in vectfit3: ==> Illegal value for opts.skip_res: {opts['skip_res']}")
        error_occurred = True
    if opts['cmplx_ss'] not in [0, 1]:
        print(f"ERROR in vectfit3_Ding: ==> Illegal value for opts.cmplx_ss: {opts['cmplx_ss']}")
        error_occurred = True

    # 如果error_occurred为True，可能需要在这里返回或中断函数执行
    if error_occurred:
        # 相应的操作，比如return或raise一个异常
        pass

    # 初始化rmserr为空列表
    rmserr = []

    # 获取s的维度，确保s是列向量
    s = s.reshape(1, -1)
    a, b = s.shape
    if a < b:
        s = s.T  # 转置s

    # 进行一些输入数组维度的基本检查
    if len(s) != len(f[0, :]):
        print('Error in vectfit3_Ding!!! ==> Second dimension of f does not match length of s.')
        return
    weight = weight.reshape(1, -1)
    if len(s) != len(weight[0, :]):
        print('Error in vectfit3_Ding!!! ==> Second dimension of weight does not match length of s.')
        return

    if weight.shape[0] != 1:
        if weight.shape[0] != f.shape[0]:
            print(
                'Error in vectfit3_Ding!!! ==> First dimension of weight is neither 1 nor matches first dimension of f.')
            return

    # 设置LAMBD为poles的对角矩阵

    # poles_1d = poles.flatten()

    # 使用 np.diag 创建一个 9x9 的对角矩阵
    LAMBD = np.diag(np.squeeze(poles))
    # LAMBD = np.diag(poles)
    Ns = len(s)
    N = len(LAMBD)
    Nc = len(f[:, 0])
    B = np.ones(N)  # 创建一个全1的列向量
    SERA = poles
    SERC = np.zeros((Nc, N))  # 初始化SERC为零矩阵
    SERD = np.zeros(Nc)  # 初始化SERD为零向量
    SERE = np.zeros(Nc)  # 初始化SERE为零向量
    roetter = poles
    fit = np.zeros((Nc, Ns))  # 初始化fit为零矩阵

    weight = weight.T  # 转置weight

    # 检查weight的维度，并设置common_weight标志
    if weight.shape[1] == 1:
        common_weight = 1
    elif weight.shape[1] == Nc:
        common_weight = 0
    else:
        print('ERROR in vectfit3_Ding: Invalid size of array weight')
        return

    # 根据opts.asymp设置offs
    if opts['asymp'] == 1:
        offs = 0
    elif opts['asymp'] == 2:
        offs = 1
    else:
        offs = 2

    if opts['skip_pole'] != 1:
        Escale = np.zeros(Nc + 1)

        # 初始化用于标记复数极点的索引数组
        cindex = np.zeros(N, dtype=int)

        # 遍历所有极点，找出是复数的极点
        for m in range(N):  # 注意Python是从0开始索引的

            if np.imag(LAMBD[m, m]) != 0:  # 检查极点是否为复数
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        if m + 1 < N:  # 防止索引越界
                            cindex[m + 1] = 2
                    else:
                        cindex[m] = 2

        Dk = np.zeros((Ns, N), dtype=complex)
        for m in range(N):  # 遍历极点
            if cindex[m] == 0:  # 实极点
                Dk[:, m] = (1. / (s - LAMBD[m, m])).flatten()
            elif cindex[m] == 1:  # 复数极点，第一部分
                Dk[:, m] = (1. / (s - LAMBD[m, m]) + 1. / (s - np.conj(LAMBD[m, m]))).flatten()
                if m+1 < N:
                    Dk[:, m + 1] = (1j / (s - LAMBD[m, m]) - 1j / (s - np.conj(LAMBD[m, m]))).flatten()

        # 根据opts.asymp的值添加列
        if opts['asymp'] == 1 or opts['asymp'] == 2:
            Dk = np.hstack((Dk, np.ones((Ns, 1))))
        elif opts['asymp'] == 3:
            Dk = np.hstack((Dk, np.ones((Ns, 1)), s.reshape(-1, 1)))

        # 最小二乘问题的缩放
        scale = 0
        for m in range(Nc):
            if weight.shape[1] == 1:
                scale += (np.linalg.norm(weight[:, 0] * f[m, :].T)) ** 2
            else:
                scale += (np.linalg.norm(weight[:, m] * f[m, :].T)) ** 2

        scale = np.sqrt(scale) / Ns

        # 构建AA和bb矩阵
        if opts['relax'] == 1:
            AA = np.zeros((Nc * (N + 1), N + 1), dtype=complex)
            bb = np.zeros(Nc * (N + 1), dtype=complex)
            Escale = np.zeros(AA.shape[1])
            for n in range(Nc):
                A = np.zeros((Ns, N + offs + N + 1), dtype=complex)
                weig = weight if common_weight == 1 else weight[:, n]
                # 左块
                for m in range(N + offs):
                    weig = weig.flatten()
                    A[:, m] = weig * Dk[:, m]
                # 右块
                inda = N + offs
                for m in range(1, N + 2):  # 修改迭代范围为从1到N+1（包括N+1）
                    A[:, inda + m - 1] = -weig * Dk[:, m - 1] * f[n, :].T  # conj()

                    # 处理复数部分
                A = np.vstack((np.real(A), np.imag(A)))

                offset = N + offs
                if n == Nc - 1:
                    # 添加一个新的行到矩阵A，新的行是一个零向量，长度与A的列数相同
                    A = np.vstack((A, np.zeros((1, A.shape[1]))))
                    for mm in range(N+1):
                        A[2 * Ns, offset + mm] = np.real(scale * np.sum(Dk[:, mm], axis=0))

                # QR分解
                Q, R = np.linalg.qr(A, mode='reduced')
                ind1 = N + offs + 1
                ind2 = N + offs + N + 1
                R22 = R[ind1 - 1:ind2, ind1 - 1:ind2]
                # 对 AA 的赋值
                AA[n * (N + 1):(n + 1) * (N + 1), :] = R22

                # 对 bb 的赋值
                if n == Nc - 1:
                    # 由于 Q 的切片是一维的，所以不需要转置操作
                    bb[n * (N + 1):(n + 1) * (N + 1)] = Q[-1, N + offs:] * Ns * scale

            # 缩放AA矩阵的列
            for col in range(AA.shape[1]):
                Escale[col] = 1 / np.linalg.norm(AA[:, col])
                AA[:, col] *= Escale[col]
            # 解线性系统
            AA = np.real(AA)
            bb = np.real(bb)
            # x = np.linalg.solve(AA, bb)
            x = np.linalg.lstsq(AA, bb, rcond=None)[0]
            x = x * Escale.T

        if opts['relax'] == 0 or abs(x[-1]) < TOLlow or abs(x[-1]) > TOLhigh:
            AA = np.zeros((Nc * N, N), dtype=complex)
            bb = np.zeros(Nc * N, dtype=complex)
            if opts['relax'] == 0:
                Dnew = 1
            else:
                if x[-1] == 0:
                    Dnew = 1
                elif abs(x[-1]) < TOLlow:
                    Dnew = np.sign(x[-1]) * TOLlow
                elif abs(x[-1]) > TOLhigh:
                    Dnew = np.sign(x[-1]) * TOLhigh

            for n in range(Nc):
                A = np.zeros((Ns, (N + offs) + N), dtype=complex)
                weig = weight[:, n] if not common_weight else weight[:, 0]

                for m in range(N + offs):  # 左块
                    A[:, m] = weig * Dk[:, m]
                inda = N + offs
                for m in range(N):  # 右块
                    A[:, inda + m] = -weig * Dk[:, m] * f[n, :].T  # conj()
                b = Dnew * weig * f[n, :].T  # conj()
                A = np.vstack((np.real(A), np.imag(A)))
                b = np.concatenate((np.real(b), np.imag(b)))
                offset = (N + offs)
                Q, R = np.linalg.qr(A, mode='reduced')
                ind1 = N + offs
                ind2 = N + offs + N
                R22 = R[ind1:ind2, ind1:ind2]
                AA[(n - 1) * N:n * N, :] = R22
                bb[(n - 1) * N:n * N] = np.dot(Q.T, b)[ind1 - 1:ind2]

            # 缩放AA矩阵的列
            Escale = np.zeros(N)
            for col in range(N):
                Escale[col] = 1 / np.linalg.norm(AA[:, col])
                AA[:, col] *= Escale[col]

            # 解线性系统
            x = np.linalg.solve(AA, bb)
            x = x * Escale.T  # x *= Escale
            x = np.append(x, Dnew)

        C = x[:-1]
        D = x[-1]

        C = np.complex128(C)
        # temp_C = np.zeros(C.shape, dtype=complex)
        # 将 C 转换回复数形式
        for m in range(N):
            if cindex[m] == 1:
                r1 = C[m]
                if m+1 < N: # python中没有自动扩充矩阵的功能，防止索引越界
                    r2 = C[m + 1]
                C[m] = r1 + 1j*r2
                if m+1 < N:
                    C[m + 1] = r1 - 1j * r2

        # N = LAMBD.shape[0]  # 假设LAMBD是一个NxN的矩阵

        m = 0
        for n in range(1, N + 1):  # Python中的循环是从1开始到N（包括N）
            m += 1
            if m < N:
                # 在Python中，数组索引从0开始，所以我们要使用 m-1 来代替代码中的 m
                if abs(LAMBD[m - 1, m - 1]) > abs(np.real(LAMBD[m - 1, m - 1])):
                    LAMBD[m, m - 1] = -np.imag(LAMBD[m - 1, m - 1])
                    LAMBD[m - 1, m] = np.imag(LAMBD[m - 1, m - 1])
                    LAMBD[m - 1, m - 1] = np.real(LAMBD[m - 1, m - 1])
                    LAMBD[m, m] = LAMBD[m - 1, m - 1]
                    B[m - 1] = 2
                    B[m] = 0
                    koko = C[m - 1]
                    C[m - 1] = np.real(koko)
                    C[m] = np.imag(koko)
                    m += 1

        # 将B和C转为列向量形式
        B = B.reshape(-1, 1)
        C = C.reshape(-1, 1)
        ZER = LAMBD - np.dot(B, C.T) / D

        # roetter = LA.eig(ZER)
        roetter = np.linalg.eigvals(np.real(ZER)).T
        unstables = np.real(roetter) > 0

        if opts['stable'] == 1:
            # 强制不稳定的极点变为稳定
            roetter[unstables] -= 2 * np.real(roetter[unstables])

        if isinstance(roetter[0], complex):
            roetter = sorted(roetter, key=abs) # 不能使用np.sort，其默认按照实部进行排序，我们需要根据模的大小进行排序
        else:
            roetter = np.sort(roetter)
        N = len(roetter)

        # 将实数极点和复数极点分开排序
        for n in range(N):
            for m in range(n + 1, N):
                if np.imag(roetter[m]) == 0 and np.imag(roetter[n]) != 0:
                    roetter[n], roetter[m] = roetter[m], roetter[n]

        N1 = 0
        for m in range(N):
            if np.imag(roetter[m]) == 0:
                N1 = m

        # 如果存在复数极点，则对它们进行排序
        if N1 < N - 1:
            if isinstance(roetter[0], complex):
                roetter[N1 + 1:] = sorted(roetter[N1 + 1:], key=abs)#np.sort(roetter[N1 + 1:])
            else:
                roetter[N1 + 1:] = np.sort(roetter[N1 + 1:])

                # 对所有极点应用转换来调整它们的虚部
        roetter = roetter - 2j * np.imag(roetter)
        roetter = roetter.reshape(-1, 1)
        # 最终转置roetter以匹配MATLAB中的行向量表示
        SERA = roetter  # SERA为9*1

    if opts['skip_res'] != 1:

        # 使用σ的修正零点（在MATLAB代码中为roetter）重新初始化LAMBD
        LAMBD = roetter

        # 初始化cindex来识别复杂极点
        cindex = [0] * N
        for m in range(N):
            if LAMBD[m].imag != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        if m + 1 < N:
                            cindex[m + 1] = 2
                    else:
                        cindex[m] = 2

        if opts['asymp'] == 1:
            A = np.zeros((2 * Ns, N), dtype=complex)
            BB = np.zeros((2 * Ns, Nc), dtype=complex)
        elif opts['asymp'] == 2:
            A = np.zeros((2 * Ns, N + 1), dtype=complex)
            BB = np.zeros((2 * Ns, Nc), dtype=complex)
        else:
            A = np.zeros((2 * Ns, N + 2), dtype=complex)
            BB = np.zeros((2 * Ns, Nc), dtype=complex)

        # 计算Dk
        Dk = np.zeros((Ns, N), dtype=complex)
        for m in range(N):
            if cindex[m] == 0:  # real pole
                Dk[:, m] = 1 / np.squeeze(s - LAMBD[m][0])
            elif cindex[m] == 1:  # complex pole, 1st part
                # acd = (1 / (s - LAMBD[m][0])) + (1 / (s - np.squeeze(np.transpose(LAMBD[m]))))
                # Dk[:, m] = (1 / (s - LAMBD[m][0])) + (1 / (s - np.squeeze(np.transpose(LAMBD[m]))))
                Dk[:, m] = np.squeeze((1 / (s - LAMBD[m][0])) + (1 / (s - np.transpose(LAMBD[m]))))  # todo有可能还要改
                if m+1 < N:
                    Dk[:, m + 1] = np.squeeze(1j / (s - LAMBD[m][0]) - 1j / (s - np.transpose(LAMBD[m])))

        # todo2 差
        if common_weight == 1:
            Dk = np.zeros((Ns, N), dtype=np.complex128)
            for m in range(N):
                if cindex[m] == 0:  # real pole
                    Dk[:, m] = np.squeeze(weight) / np.squeeze(s - LAMBD[m][0])
                elif cindex[m] == 1:  # complex pole, 1st part
                    Dk[:, m] = np.squeeze(weight / (s - np.conj(LAMBD[m])) + weight / (s - LAMBD[m])) # LAMBD已经转置过，此处与matlab计算不同
                    if m + 1 < N:
                        Dk[:, m + 1] = np.squeeze(1j * weight / (s - np.conj(LAMBD[m])) - 1j * weight / (s - LAMBD[m]))# LAMBD已经转置过，此处与matlab计算不同

            # 更新A和BB矩阵
            if opts['asymp'] == 1:
                A[:Ns, :N] = Dk
            elif opts['asymp'] == 2:
                A[:Ns, :N] = Dk
                A[:Ns, N] = weight
            else:
                A[:Ns, :N] = Dk
                A[:Ns, N] = np.squeeze(weight)
                A[:Ns, N + 1] = np.squeeze(weight * s)

            for m in range(Nc):
                BB[:Ns, m] = np.squeeze(weight) * f[m, :]

            # 1分离实部和虚部
            A[Ns:2 * Ns, :] = np.imag(A[0:Ns, :])
            A[0:Ns, :] = np.real(A[0:Ns, :])
            BB[Ns:2 * Ns, :] = np.imag(BB[0:Ns, :])
            BB[0:Ns, :] = np.real(BB[0:Ns, :])

            if opts['asymp'] == 2:
                A[:Ns, N] = A[:Ns, N]
            elif opts['asymp'] == 3:
                A[:Ns, N] = A[:Ns, N]
                A[Ns:2 * Ns, N + 1] = A[Ns:2 * Ns, N + 1]
            # # 归一化A列
            # Escale = np.linalg.norm(A, axis=0)
            # A = A / Escale
            #
            # # 解线性方程组
            # X = np.dot(np.linalg.pinv(A), BB)  # X = np.linalg.solve(A, BB)
            # 假设A和BB已经被适当初始化

            Escale = np.zeros(A.shape[1])
            for col in range(A.shape[1]):
                Escale[col] = np.linalg.norm(A[:, col], 2)
                A[:, col] = A[:, col] / Escale[col]

            X = np.dot(np.linalg.pinv(A), BB)

            # 归一化X
            for n in range(Nc):
                X[:, n] /= Escale

            # 转置X以符合MATLAB输出格式
            X = X.T

            # 提取C, SERD, SERE
            C = X[:, 0:N]
            SERD = np.zeros(Nc)
            SERE = np.zeros(Nc)
            if opts['asymp'] == 2:
                SERD = X[:, N]
            elif opts['asymp'] == 3:
                SERD = X[:, N]
                SERE = X[:, N + 1]


        for m in range(N):
            if cindex[m] == 1:
                for n in range(Nc):
                    r1 = C[n, m]
                    r2 = C[n, m + 1]
                    C[n, m] = r1 + 1j * r2
                    C[n, m + 1] = r1 - 1j * r2

        B = np.ones((N, 1))

        # 设置SERA, SERB, SERC
        SERA = LAMBD
        SERB = B
        SERC = C

        # 31计算拟合结果
        Dk = np.zeros((Ns, N), dtype=complex)
        for m in range(N):
            Dk[:, m] = 1.0 / np.squeeze(s - SERA[m][0])

        # 计算 fit
        fit = np.zeros((Nc, Ns), dtype=complex)
        for n in range(Nc):
            fit[n, :] = np.dot(Dk, SERC[n, :].T).T
            if opts['asymp'] == 2:
                fit[n, :] += SERD[n]
            elif opts['asymp'] == 3:
                fit[n, :] += np.squeeze(SERD[n] + np.dot(s, SERE[n]))

        # 计算RMS误差
        fit = fit.T
        diff = fit - f.T
        rmserr = np.sqrt(np.sum(np.abs(diff ** 2))) / np.sqrt(Nc * len(s))

        fit = fit.T

    #######################
    A = SERA
    poles = A
    if opts['skip_res'] != 1:  # 假设 opts_skip_res 对应于 MATLAB 中的 opts.skip_res
        B = SERB
        C = SERC
        D = SERD
        E = SERE
    else:
        B = np.ones((N, 1))
        C = np.zeros((Nc, N))
        D = np.zeros((Nc, Nc))  # 注意：MATLAB中D和E的初始化为(Nc,Nc)，但通常D为直流项，应为一维
        E = np.zeros((Nc, Nc))  # 根据实际情况，这里假设D和E应为(Nc, 1)，如需不同，请调整
        rmserr = 0

    if opts['cmplx_ss'] != 1:
        # 将A转换为对角矩阵形式
        A = np.diag(A)

        N = len(A)  # 状态的数量
        cindex = np.zeros(N)

        # 标记复数元素
        for m in range(N):
            if np.imag(A[m]) != 0:
                if m == 0 or (m > 0 and (cindex[m - 1] == 0 or cindex[m - 1] == 2)):
                    cindex[m] = 1
                    if m + 1 < N:
                        cindex[m + 1] = 2
                else:
                    cindex[m] = 2

        m = 0
        while m < N:
            if cindex[m] == 1:
                a = A[m]
                a1, a2 = np.real(a), np.imag(a)
                c = C[:, m]
                c1, c2 = np.real(c), np.imag(c)
                b = B[m]
                b1, b2 = 2 * np.real(b), -2 * np.imag(b)
                Ablock = np.array([[a1, a2], [-a2, a1]])

                A[m:m + 2, m:m + 2] = Ablock
                C[:, m] = c1
                C[:, m + 1] = c2
                B[m] = b1
                B[m + 1] = b2
                m += 1  # Skip next column, already processed as part of complex pair
            m += 1
    else:
        # A保持为复数对角矩阵
        A = np.diag(A.flatten())  # A = np.diag(np.diag(A))  # 确保A是对角矩阵

    # 假设A, B, C, D, E是已经计算好的变量
    if D.ndim == 1:
        D = D.reshape(1, -1)
    if E.ndim == 1:
        E = E.reshape(1, -1)
    SER = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "E": E
    }

    return SER, poles, rmserr, fit
