import numpy as np
import time
from scipy.linalg import eig
from Vector_Fitting.Calculators.fit_vector import vectfit3
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

#135行     Ns = len(s)  改为  Ns = s.size
#146行     f = np.zeros((Nc * (Nc + 1) // 2, len(bigH)), dtype=complex) 变为 f = np.zeros((Nc * (Nc + 1) // 2, bigH.size), dtype=complex)
#93行改了
#76行
# 506行


# 【函数说明】
# 【参数说明】
# - 入参：bigH（n*n*n的复数矩阵），s（1*n的复数list），poles（初始是空矩阵列表）, opts字典映射结构体
# - 出参：SER, rmserr, bigHfit, opts
# 【功能说明】
def drive(bigH, s, poles, opts):
    # Default settings
    def_opts = {
        'N': [],
        'poletype': 'lincmplx',
        'nu': 1e-3,
        'Niter1': 4,
        'Niter2': 4,
        'weight': [],
        'weightparam': 1,
        'asymp': 2,
        'stable': 1,
        'relaxed': 1,
        'plot': 1,
        'logx': 1,
        'logy': 1,
        'errplot': 1,
        'phaseplot': 1,
        'screen': 1,
        'cmplx_ss': 1,
        'remove_HFpoles': 0,
        'factor_HF': 1.1,
        'passive_DE': 0,
        'passive_DE_TOLD': 1e-6,
        'passive_DE_TOLE': 1e-16
    }

    if opts is None:
        opts = def_opts
    else:
        # Merge default values into opts
        for key, value in def_opts.items():
            if key not in opts:
                opts[key] = value

    VF = {
        'asymp': opts['asymp'],
        'stable': 1 if opts['stable'] == 1 else 0,
        'relax': 1 if opts['relaxed'] == 1 else 0,
        'spy2': 1 if opts['plot'] == 1 else 0,
        'logx': 1 if opts['logx'] == 1 else 0,
        'logy': 1 if opts['logy'] == 1 else 0,
        'errplot': 1 if opts['errplot'] == 1 else 0,
        'phaseplot': 1 if opts['phaseplot'] == 1 else 0,
        'cmplx_ss': 1 if opts['cmplx_ss'] == 1 else 0
    }

    Niter1 = opts['Niter1']
    Niter2 = opts['Niter2']
    weightparam = opts['weightparam']
    remove_HFpoles = opts['remove_HFpoles']
    factor_HF = opts['factor_HF']
    passive_DE = opts['passive_DE']
    passive_DE_TOLD = opts['passive_DE_TOLD']
    passive_DE_TOLE = opts['passive_DE_TOLE']

    fit1, fit2, fit3 = [], [], []

    Ns = s.size

    if not poles:
        if not opts['N']:
            print('===> ERROR in poleresiduefit.m: You did not specify a value for opts.N (fitting order). Must stop.')
            print('     Solution: either specify value for opts.N, or provide initial poles in array poles.')
            return
        
        N = opts['N']
        oldpoletype = opts['poletype']
        if N < 6:
            if opts['poletype'] == 'linlogcmplx':
                opts['poletype'] = 'logcmplx'

        nu = opts['nu']
        if len(opts['poletype']) == 8:
            if opts['poletype'] == 'lincmplx':  # Complex, linearly spaced starting poles
                bet = np.linspace(np.squeeze(s)[0] / 1j, np.squeeze(s)[Ns - 1] / 1j, int(N / 2))
                poles = []
                for n in range(len(bet)):
                    alf = -nu * bet[n]
                    poles.extend([(alf - 1j * bet[n]), (alf + 1j * bet[n])])
            elif opts['poletype'] == 'logcmplx':  # Complex, logarithmically spaced starting poles
                bet = np.logspace(np.log10(s[0] / 1j), np.log10(s[Ns - 1] / 1j), int(N / 2))
                poles = []
                for n in range(len(bet)):
                    alf = -nu * bet[n]
                    poles.extend([(alf - 1j * bet[n]), (alf + 1j * bet[n])])
            else:
                print('-->ERROR in poleresiduefit.m: Illegal value for opts.poletype')
                print('   Valid input: \'lincmplex\' and \'logcmplx\'')
                print('   Given input:')
                print(opts['poletype'])
                return
        elif len(opts['poletype']) == 11:
            if opts['poletype'] == 'linlogcmplx':
                bet = np.linspace(s[0] / 1j, s[Ns - 1] / 1j, int((N - 1) / 4) + 1)
                poles1 = []
                for n in range(len(bet)):
                    alf = -nu * bet[n]
                    poles1.extend([(alf - 1j * bet[n]), (alf + 1j * bet[n])])
                bet = np.logspace(np.log10(s[0] / 1j), np.log10(s[Ns - 1] / 1j), int(N / 4) + 2)
                bet = bet[1:-1]
                poles2 = []
                for n in range(len(bet)):
                    alf = -nu * bet[n]
                    poles2.extend([(alf - 1j * bet[n]), (alf + 1j * bet[n])])
                poles = poles1 + poles2
            else:
                print('-->ERROR in poleresiduefit.m: Illegal value for opts.poletype')
                print('   Valid input: \'lincmplex\' and \'logcmplx\'')
                print('   Given input:')
                print(opts['poletype'])
                return

        if len(poles) < N:  # An odd number of poles was prescribed
            if opts['poletype'] == 'lincmplx':
                pole_extra = -(np.squeeze(s)[0] / 1j + np.squeeze(s)[Ns - 1] / 1j) / 2  # Placing surplus pole in midpoint
            elif opts['poletype'] == 'logcmplx' or opts['poletype'] == 'linlogcmplx':
                pole_extra = -10 ** ((np.log10(s[0] / 1j) + np.log10(s[Ns - 1] / 1j)) / 2)  # Placing surplus pole in midpoint
            poles.append(pole_extra)
        opts['poletype'] = oldpoletype

    Nc = bigH.shape[0]
    Ns = s.size
    
    if opts['screen'] == 1:
        
        start_time = time.time()
        print('-----------------S T A R T--------------------------')

    if opts['screen'] == 1:
        print('****Stacking matrix elements (lower triangle) into single column ...')

    tell = 0
    f = np.zeros((Nc * (Nc + 1) // 2, bigH.size), dtype=complex)
    for col in range(Nc):
        for row in range(col, Nc):
            tell += 1
            f[tell - 1, :] = np.squeeze(bigH[row, col, :])

    nnn = tell

    # Fitting options
    VF['spy1'] = 0
    VF['skip_pole'] = 0
    VF['skip_res'] = 1
    VF['legend'] = 1
    oldspy2 = VF['spy2']
    VF['spy2'] = 0

    if Nc == 1:
        f_sum = f
    else:  # Will do only for multi-terminal case
        # Forming columns sum and associated LS weight:
        f_sum = np.zeros((1, Ns), dtype=complex)
        tell = 0
        for row in range(Nc):
            for col in range(row, Nc):
                tell += 1
                if weightparam == 1 or weightparam == 4 or weightparam == 5:
                    f_sum += f[tell - 1, :]  # unweighted sum
                elif weightparam == 2:
                    f_sum += f[tell - 1, :] / np.linalg.norm(f[tell - 1, :])
                elif weightparam == 3:
                    f_sum += f[tell - 1, :] / np.sqrt(np.linalg.norm(f[tell - 1, :]))

    # Creating LS weight
    if not opts['weight']:  # Automatic specification of weight
        if weightparam == 1:  # weight=1 for all elements in LS problem, at all freq.
            weight = np.ones((1, Ns))
            weight_sum = np.ones((1, Ns))
        elif weightparam == 2:  # Individual element weighting
            weight = 1.0 / np.abs(f)
            weight_sum = 1.0 / np.abs(f_sum)
        elif weightparam == 3:  # Individual element weighting
            weight = 1.0 / np.sqrt(np.abs(f))
            weight_sum = 1.0 / np.sqrt(np.abs(f_sum))
        elif weightparam == 4:  # Common weighting for all matrix elements
            weight = np.array([1.0 / np.linalg.norm(f[:, k]) for k in range(Ns)])
            weight_sum = weight
        elif weightparam == 5:  # Common weighting for all matrix elements
            weight = np.array([1.0 / np.sqrt(np.linalg.norm(f[:, k])) for k in range(Ns)])
            weight_sum = weight
        else:
            print('-->ERROR in mtrxVectfit: Illegal value for opts.weight')
            return
    else:
        weight = np.zeros((nnn, Ns))
        tell = 0
        for row in range(Nc):
            for col in range(row, Nc):
                tell += 1
                weight[tell - 1, :] = np.squeeze(opts['weight'][row, col, :])
        weight_sum = np.ones((1, Ns))

    if Nc > 1:  # Will do only for multi-terminal case
        if opts['screen'] == 1:
            print('****Calculating improved initial poles by fitting column sum ...')
        for iter in range(Niter1):
            if opts['screen'] == 1:
                print(f'Iter {iter + 1}')
            SER, poles, rmserr, fit = vectfit3(f_sum, s, poles, weight_sum, VF)

    if opts['screen'] == 1:
        print('****Fitting column ...')
    VF['skip_res'] = 1
    for iter in range(Niter2):
        if opts['screen'] == 1:
            print(f'   Iter {iter + 1}')
        if iter == Niter2 - 1:
            VF['skip_res'] = 0
        SER, poles, rmserr, fit1 = vectfit3(f, s, poles, weight, VF)
    if Niter2 == 0:
        VF['skip_res'] = 0
        VF['skip_pole'] = 1
        SER, poles, rmserr, fit1 = vectfit3(f, s, poles, weight, VF)

    # Throwing out high-frequency poles:
    fit2 = fit1
    if remove_HFpoles == 1:
        if opts['screen'] == 1:
            print('****Throwing out high-frequency poles: ...')
        ind = np.abs(poles) > factor_HF * np.abs(s[-1])
        poles = np.delete(poles, ind)  # Deleting poles above upper frequency limit
        N = len(poles)
        if opts['screen'] == 1:
            print('****Refitting residues: ...')
        VF['skip_pole'] = 1
        SER, poles, rmserr, fit2 = vectfit3(fit1, s, poles, weight, VF)

    if passive_DE == 1 and VF['asymp'] > 1:
        if opts['screen'] == 1:
            if VF['asymp'] == 2:
                print('****Enforcing positive realness for D...')
            elif VF['asymp'] == 3:
                print('****Enforcing positive realness for D, E...')
        tell = 0
        DD = np.zeros((Nc, Nc))
        EE = np.zeros((Nc, Nc))
        for col in range(Nc):
            for row in range(col, Nc):
                tell += 1
                DD[row, col] = SER['D'][tell - 1]
                EE[row, col] = SER['E'][tell - 1]
        DD = DD + DD.T - np.diag(DD.diagonal())
        EE = EE + EE.T - np.diag(EE.diagonal())

        # Calculating Dmod, Emod:
        V, L = eig(DD)
        for n in range(Nc):
            if L[n, n] < 0:
                L[n, n] = passive_DE_TOLD
        DD = V @ np.diag(L) @ np.linalg.inv(V)

        V, L = eig(EE)
        for n in range(Nc):
            if L[n, n] < 0:
                L[n, n] = passive_DE_TOLE
        EE = V @ np.diag(L) @ np.linalg.inv(V)

        tell = 0
        # Calculating fmod:
        Emod = np.zeros((Nc, 1))
        Dmod = np.zeros((Nc, 1))
        for col in range(Nc):
            for row in range(col, Nc):
                tell += 1
                Dmod[tell - 1] = DD[row, col]
                Emod[tell - 1] = EE[row, col]
                fmod[tell - 1, :] = fit2[tell - 1, :] - Dmod[tell - 1] - s * Emod[tell - 1]

        if opts['screen'] == 1:
            if VF['asymp'] == 2:
                print('****Refitting C while enforcing D=0...')
            elif VF['asymp'] == 3:
                print('****Refitting C while enforcing D=0, E=0...')

        VF['skip_pole'] = 0
        VF['asymp'] = 1
        for iter in range(1):
            # [SER, poles, rmserr, fit3] = vectfit2(fmod, s, poles, weight, VF)
            SER, poles, rmserr, fit3 = vectfit3(fmod, s, poles, weight, VF)

        SER['D'] = Dmod
        SER['E'] = Emod
        for tell in range(len(fit3[:, 0])):
            fit3[tell, :] = fit3[tell, :] + SER['D'][tell] + s * SER['E'][tell]

    if Nc > 1:
        if opts['screen'] == 1:
            # toc
            print('****Transforming model of lower matrix triangle into state-space model of full matrix ...')
        SER = tri2full(SER)

    if opts['screen'] == 1:
        print('****Generating pole-residue model ...')
    R, a = ss2pr(SER['A'], SER['B'], SER['C'])
    SER['R'] = R
    SER['poles'] = a

    # rmserror of fitting:
    if len(fit3) != 0:
        fit = fit3
    elif len(fit2) != 0:
        fit = fit2
    elif len(fit1) != 0:
        fit = fit1

    diff = fit - f

    rmserr = np.sqrt(np.sum(np.sum(np.abs(diff ** 2)))) / np.sqrt(nnn * Ns)

    VF['spy2'] = oldspy2
    if VF['spy2'] == 1:
        if opts['screen'] == 1:
            print('****Plotting of results ...')

        freq = s / (2 * np.pi * 1j)
        if VF['logx'] == 1:
            if VF['logy'] == 1:
                plt.figure(1)
                h1 = plt.loglog(freq, np.abs(f), 'b')
                plt.xlim([freq[0], freq[-1]])
                plt.hold(True)
                h2 = plt.loglog(freq, np.abs(fit), 'r--')
                plt.hold(False)
                if VF['errplot'] == 1:
                    plt.hold(True)
                    h3 = plt.loglog(freq, np.abs(f - fit), 'g')
                    plt.hold(False)
            else:  # logy=0
                plt.figure(1)
                h1 = plt.semilogx(freq, np.abs(f), 'b')
                plt.xlim([freq[0], freq[-1]])
                plt.hold(True)
                h2 = plt.semilogx(freq, np.abs(fit), 'r--')
                plt.hold(False)
                if VF['errplot'] == 1:
                    plt.hold(True)
                    h3 = plt.semilogx(freq, np.abs(f - fit), 'g')
                    plt.hold(False)
            if VF['phaseplot'] == 1:
                plt.figure(2)
                h4 = plt.semilogx(freq, 180 * np.unwrap(np.angle(f)) / np.pi, 'b')
                plt.xlim([freq[0], freq[-1]])
                plt.hold(True)
                h5 = plt.semilogx(freq, 180 * np.unwrap(np.angle(fit)) / np.pi, 'r--')
                plt.hold(False)
        else:  # logx=0
            if VF['logy'] == 1:
                plt.figure(1)
                h1 = plt.semilogy(freq, np.abs(f), 'b')
                plt.xlim([freq[0], freq[-1]])
                plt.hold(True)
                h2 = plt.semilogy(freq, np.abs(fit), 'r--')
                plt.hold(False)
                if VF['errplot'] == 1:
                    plt.hold(True)
                    h3 = plt.semilogy(freq, np.abs(f - fit), 'g')
                    plt.hold(False)
            else:  # logy=0
                plt.figure(1)
                h1 = plt.plot(freq, np.abs(f), 'b')
                plt.xlim([freq[0], freq[-1]])
                plt.hold(True)
                h2 = plt.plot(freq, np.abs(fit), 'r--')
                plt.hold(False)
                if VF['errplot'] == 1:
                    plt.hold(True)
                    h3 = plt.plot(freq, np.abs(f - fit), 'g')
                    plt.hold(False)
            if VF['phaseplot'] == 1:
                plt.figure(2)
                h4 = plt.plot(freq, 180 * np.unwrap(np.angle(f)) / np.pi, 'b')
                plt.xlim([freq[0], freq[-1]])
                plt.hold(True)
                h5 = plt.plot(freq, 180 * np.unwrap(np.angle(fit)) / np.pi, 'r--')
                plt.hold(False)
        plt.figure(1)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [p.u.]')
        plt.title('Approximation of f')
        if VF['legend'] == 1:
            if VF['errplot'] == 1:
                plt.legend([h1[0], h2[0], h3[0]], ['Original', 'FRVF', 'Deviation'])
            else:
                plt.legend([h1[0], h2[0]], ['Original', 'FRVF'])
        if VF['phaseplot'] == 1:
            plt.figure(2)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase angle [deg]')
            plt.title('Approximation of f')
            if VF['legend'] == 1:
                plt.legend([h4[0], h5[0]], ['Original', 'FRVF'])
        plt.draw()
        plt.pause(0.001)

    if opts['screen'] == 1:
        print('-------------------E N D----------------------------')

    bigHfit = np.zeros((Nc, Nc, Ns), dtype=complex)
    tell = 0
    for row in range(Nc):
        for col in range(row, Nc):
            tell += 1
            # bigHfit[row, col, :] = np.real(fit[tell - 1, :])
            bigHfit[row, col, :] = fit[tell - 1, :]
            if row != col:
                # bigHfit[col, row, :] = np.real(fit[tell - 1, :])
                bigHfit[col, row, :] = fit[tell - 1, :]
    return SER, rmserr, bigHfit, opts


def tri2full(SER):
    A = SER['A']
    B = SER['B']
    C = SER['C']
    D = SER['D']
    E = SER['E']

    tell = 0
    for k in range(1, 10001):
        tell += k
        if tell == len(D):
            Nc = k
            break

    N = len(A)
    tell = 0
    CC = np.zeros((Nc, Nc * N))
    AA = np.array([[]]).reshape(0, 0)  # 初始化为空矩阵
    BB = np.array([[]]).reshape(0, 0)  # 同上
    DD = np.zeros((Nc, Nc))
    EE = np.zeros((Nc, Nc))
    for col in range(1, Nc + 1):
        AA = block_diag(AA, A)
        BB = block_diag(BB, B)

        for row in range(col, Nc + 1):
            tell += 1
            DD[row-1, col-1] = np.real(D[tell-1])  # np.real() can select real value in complex value.
            EE[row-1, col-1] = np.real(E[tell-1])
            CC[row-1, (col-1)*N:(col*N)] = np.real(C[tell-1, :])
            CC[col-1, (row-1)*N:(row*N)] = np.real(C[tell-1, :])

    DD = DD + (DD - np.diag(np.diag(DD))).T
    EE = EE + (EE - np.diag(np.diag(EE))).T

    SER2 = {'A': AA, 'B': BB.T, 'C': CC, 'D': DD, 'E': EE}
    return SER2


def ss2pr(A, B, C):
    """
    Convert state-space model having COMMON POLE SET into pole-residue model.

    Input:
    A, B, C: must have the format produced by vectfit2.m. Both
             formats determined by parameter VF.cmplx_ss are valid

    Output:
    R (Nc, Nc, N): Residues
    a (N): poles

    This routine is part of the vector fitting package (v2.1)
    Last revised: 27.10.2005.
    Created by: Bjorn Gustavsen.
    """

    # Converting real-only state-space model into complex model, if necessary
    if np.max(np.abs(A - np.diag(np.diag(A)))) != 0:
        errflag = 0
        for m in range(len(A) - 1):
            if A[m, m + 1] != 0:
                A[m, m] = A[m, m] + 1j * A[m, m + 1]
                A[m + 1, m + 1] = A[m + 1, m + 1] - 1j * A[m, m + 1]

                B[m, :] = (B[m, :] + B[m + 1, :]) / 2
                B[m + 1, :] = B[m, :]

                C[:, m] = C[:, m] + 1j * C[:, m + 1]
                C[:, m + 1] = np.conj(C[:, m])

    # Converting complex state-space model into pole-residue model
    Nc = len(C[:, 0])
    N = len(A) // Nc
    R = np.zeros((Nc, Nc, N), dtype=complex)
    for m in range(N):
        Rdum = np.zeros((Nc, Nc), dtype=complex)
        for n in range(Nc):
            ind = n * N + m
            if B.shape == C.shape:
                B = B.T
            Rdum += np.outer(C[:, ind], B[ind, :])
        R[:, :, m] = Rdum

    a = np.diag(A[:N, :N])

    return R, a

# 71行len改为np.shape