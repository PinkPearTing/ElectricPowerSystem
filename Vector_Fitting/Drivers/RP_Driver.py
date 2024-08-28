import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from time import time
from Vector_Fitting.Calculators.fitcalcABCDE import fitcalcABCDE
from Vector_Fitting.Calculators.intercheig import intercheig
from Vector_Fitting.Calculators.pr2ss import pr2ss
from Vector_Fitting.Calculators.rot import rot
from Vector_Fitting.Calculators.violextremaY import violextremaY
from Vector_Fitting.Calculators.FRPY import FRPY

# 482行
# 483行
#482 536 558 559
# 419 I = ...  # 422-424  #418 Ns

def drive(SER, s, opts=None):
    #--------------------------------------------
    MPopts = {
        'auxflag': 1,
        'solver': 'QUADPROG'
    }
    #---------------------------------------------

    SER = pr2ss(SER)  # Convert model from pole-residue to state-space

    print('-----------------S T A R T--------------------------')

    if opts is None:
        opts = {}

    if 'parametertype' not in opts:
        opts['parametertype'] = 'Y'
    if 'Niter_out' not in opts:
        opts['Niter_out'] = 10
    if 'Niter_in' not in opts:
        opts['Niter_in'] = 0
    if 'TOLGD' not in opts:
        opts['TOLGD'] = 1e-6
    if 'TOLE' not in opts:
        opts['TOLE'] = 1e-12
    if 'cmplx_ss' not in opts:
        opts['cmplx_ss'] = 1
    if 'weightfactor' not in opts:
        opts['weightfactor'] = 0.001
    if 'weightparam' not in opts:
        opts['weightparam'] = 1
    if 'method' not in opts:
        opts['method'] = 'FRP'
    if 'colinterch' not in opts:
        opts['colinterch'] = 1
    if 'outputlevel' not in opts:
        opts['outputlevel'] = 1
    if 'weight' not in opts:
        opts['weight'] = []

    if opts['method'] == 'FMP' and opts['parametertype'] == 'S':
        # disp('ERROR in RPdriver.m: FMP cannot be used together with S-parameters. Must stop.')
        # return
        pass

    colinterch = opts['colinterch']

    MPopts['TOLGD'] = opts['TOLGD']
    MPopts['TOLE'] = opts['TOLE']
    MPopts['weightfactor'] = opts['weightfactor']
    MPopts['weightparam'] = opts['weightparam']
    MPopts['weight'] = opts['weight']
    MPopts['outputlevel'] = opts['outputlevel']

    opts2 = {
        'method': opts['method'],
        'parametertype': opts['parametertype'],
        'Niter_out': opts['Niter_out'],
        'Niter_in': opts['Niter_in'],
        'TOLGD': opts['TOLGD'],
        'TOLE': opts['TOLE'],
        'cmplx_ss': opts['cmplx_ss'],
        'weightparam': opts['weightparam'],
        'weightfactor': opts['weightfactor'],
        'colinterch': opts['colinterch'],
        'outputlevel': opts['outputlevel']
    }

    if opts['parametertype'] == 'Y':
        print('*** Y-PARAMETERS ***')
    elif opts['parametertype'] == 'S':
        print('*** S-PARAMETERS ***')

    plotte = 0
    if 'plot' in opts:
        if opts['plot']:
            plotte = 1
            s_pass = opts['plot']['s_pass']  # For plotting purposes
            xlimflag = 'xlim' in opts['plot']
            ylimflag = 'ylim' in opts['plot']

    break_outer = 0
    olds3 = []

    SER0 = SER
    Nc = len(SER['D'])

    Niter_out = opts['Niter_out']
    Niter_in = opts['Niter_in']

    #========================================================
    # Plotting eigenvalues of original model (SERC0, SERD0):
    #========================================================
    if plotte == 1:
        oldT0 = []
        oldU = []
        I = sp.eye(len(SER['A'][:, 0]))
        for k in range(len(s_pass)):
            Y = SER['C'] @ np.diag((s_pass[k] * I - np.diag(SER['A'])) ** (-1)) @ SER['B'] + SER['D'] + s_pass[k] * SER['E']
            if opts['parametertype'] == 'Y':
                G = np.real(Y)
                T0, D = la.eig(G)
                T0 = rot(T0)  # minimizing phase angle of eigenvectors in least squares sense
                T0, D = intercheig(T0, oldT0, D, Nc, k)
                oldT0 = T0
                EE0[:, k] = np.diag(D)
            elif opts['parametertype'] == 'S':
                if colinterch == 0:
                    EE0[:, k] = la.svd(Y, compute_uv=False)
                else:
                    U, S, V = la.svd(Y, full_matrices=False)
                    U, S, V = interchsvd(U, oldU, S, V, Nc, k)
                    oldU = U
                    EE0[:, k] = np.diag(S)

        plt.figure(7)
        h0 = plt.plot(s_pass / (2 * np.pi * 1j), EE0.T, 'b')
        plt.hold(True)
        plt.grid(True)
        if xlimflag:
            plt.xlim(opts['plot']['xlim'])
        else:
            plt.xlim([s_pass[0] / (2 * np.pi * 1j), s_pass[-1] / (2 * np.pi * 1j)])
        if ylimflag:
            plt.ylim(opts['plot']['ylim'])
        plt.xlabel('Frequency [Hz]')
        if opts['parametertype'] == 'Y':
            plt.ylabel('Eigenvalues of G_{rat}')
        else:
            plt.ylabel('Singular values of S_{rat}')

        plt.figure(8)
        h2 = plt.plot(s_pass / (2 * np.pi * 1j), EE0.T, 'b')
        plt.hold(True)
        plt.grid(True)
        if xlimflag:
            plt.xlim(opts['plot']['xlim'])
        else:
            plt.xlim([s_pass[0] / (2 * np.pi * 1j), s_pass[-1] / (2 * np.pi * 1j)])
        if ylimflag:
            plt.ylim(opts['plot']['ylim'])
        plt.title('Monitoring enforcement process (eig(G(s))')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True)
        plt.pause(0.1)

    outputlevel = opts['outputlevel']
    t = [0, 0, 0, 0]
    tdumtotal = time()

    #=======================================================
    # Passivity Enforcement :
    #=======================================================
    QP = {'first': 1}
    QPopts = []

    SER1 = SER0

    for iter_out in range(Niter_out):
        if break_outer == 1:
            SER0 = SER1
            break
        s3 = []  # Initial value

        for iter_in in range(1, Niter_in + 2):
            s2 = []
            SERflag = 1
            if outputlevel == 1:
                print(' ')
                print(f'  [ {iter_out}  {iter_in - 1} ]')
                print('  Passivity Assessment:')
            tdum = time()

            if iter_in == 1:
                if opts['parametertype'] == 'Y':
                    wintervals = pass_check_Y(SERflag, SER['poles'], [], SER1['R'], SER1['D'], colinterch)
                else:
                    TOL = 1e-3
                    spy = 1
                    wintervals = pass_check_S(SERflag, SER['poles'], [], SER1['R'], SER1['D'], TOL, spy, colinterch)
                t[0] += time() - tdum

                if len(wintervals) > 0:
                    if outputlevel == 1:
                        print(f'    N.o. violating intervals: {len(wintervals[0, :])}')

                if opts['parametertype'] == 'Y':
                    if len(wintervals) == 0 and np.all(la.eig(SER1['D'])[0] >= 0) and np.all(la.eig(SER1['E'])[0] >= 0):
                        SER0 = SER1
                        break_outer = 1
                        break
                elif opts['parametertype'] == 'S':
                    if len(wintervals) == 0 and np.all(la.svd(SER1['D']) <= 1):
                        SER0 = SER1
                        break_outer = 1
                        break

                if opts['parametertype'] == 'Y':
                    tdum = time()
                    s_viol, g_pass, ss = violextremaY(SERflag, wintervals.T, SER['poles'], [], SER1['R'], SER1['D'], colinterch)
                    t[1] += time() - tdum
                    s2 = s_viol.T
                    s2 = np.sort(s2)
                    if len(s2) == 0 and np.all(la.eig(SER1['D'])[0] > 0):
                        break
                elif opts['parametertype'] == 'S':
                    tdum = time()
                    s_viol, g_pass, ss = violextremaS(SERflag, wintervals.T, SER['poles'], [], SER1['R'], SER1['D'], colinterch)
                    t[1] += time() - tdum
                    s2 = s_viol.T
                    s2 = np.sort(s2)
                    SER1 = SER0
                    if len(s2) == 0 and np.all(la.svd(SER1['D']) < 1):
                        break

            if iter_in == 1:
                if outputlevel == 1:
                    if opts['parametertype'] == 'Y':
                        if np.min(g_pass) < 0:
                            print(f'    Max. violation, eig(G) : {g_pass} @ {(ss) / (2 * np.pi * 1j)} Hz')
                        else:
                            print('    Max. violation, eig(G) :  None')
                        if np.min(la.eig(SER0['D'])[0]) < 0:
                            print(f'    Max. violation, eig(D) : {np.min(la.eig(SER1["D"])[0])}')
                        else:
                            print('    Max. violation, eig(D) :  None')
                        if np.min(la.eig(SER0['E'])[0]) < 0:
                            print(f'    Max. violation, eig(E) : {np.min(la.eig(SER1["E"])[0])}')
                        else:
                            print('    Max. violation, eig(E) :  None')
                    elif opts['parametertype'] == 'S':
                        if np.max(g_pass) > 1:
                            print(f'    Max. violation, sing(S) : {-1 + g_pass} @ {(ss) / (2 * np.pi * 1j)} Hz')
                        else:
                            print('    Max. violation, sing(S) :  None')
                        if np.max(la.svd(SER0['D'])) > 1:
                            print(f'    Max. violation, sing(D) : {-1 + np.max(la.svd(SER1["D"]))}')
                        else:
                            print('    Max. violation, sing(D) :  None')
                if outputlevel != 1:
                    if opts['parametertype'] == 'Y':
                        min1 = np.min(g_pass)
                        min2 = np.min(la.eig(SER1['D'])[0])
                        print(f'    Max. violation  : {np.min([min1, min2])}')
                        if np.min(la.eig(SER0['E'])[0]) < 0:
                            print(f'    Max. violation, E: {np.min(la.eig(SER1["E"])[0])}')
                    elif opts['parametertype'] == 'S':
                        max1 = np.max(g_pass)
                        max2 = np.max(la.eig(SER1['D'])[0])
                        print(f'    Max. violation  : {-1 + np.max([max1, max2])}')

            if len(s3) > 0:
                pass

            if outputlevel == 1:
                print('  Passivity Enforcement...')
            tdum = time()
            if opts['method'] == 'FMP':
                SER1, MPopts = FMP(SER0, s, s2, s3, MPopts)
            elif opts['method'] == 'FRP':
                if opts['parametertype'] == 'Y':
                    SER1, MPopts = FRPY(SER0, s, s2, s3, MPopts)
                else:
                    SER1, MPopts = FRPS(SER0, s, s2, s3, MPopts)
            else:
                print('****** ERROR #1 in FMPdriver.m')
            t[2] += time() - tdum

            t[3] += time() - tdumtotal
            if plotte == 1:
                if opts['parametertype'] == 'Y':
                    oldT0 = []
                    tell = 0
                    I = sp.eye(len(SER['A'][:, 0]))
                    for k in range(len(s_pass)):
                        Y = SER1['C'] @ np.diag((s_pass[k] * I - np.diag(SER1['A'])) ** (-1)) @ SER1['B'] + SER1['D'] + s_pass[k] * SER1['E']
                        G = np.real(Y)
                        T0, D = la.eig(G)
                        T0 = rot(T0)  # minimizing phase angle of eigenvectors in least squares sense
                        T0, D = intercheig(T0, oldT0, D, Nc, k)
                        oldT0 = T0
                        EE1[:, k] = np.diag(D)
                elif opts['parametertype'] == 'S':
                    oldU = []
                    tell = 0
                    I = sp.eye(len(SER['A'][:, 0]))
                    for k in range(len(s_pass)):
                        Y = SER1['C'] @ np.diag((s_pass[k] * I - np.diag(SER1['A'])) ** (-1)) @ SER1['B'] + SER1['D'] + s_pass[k] * SER1['E']
                        if colinterch == 0:
                            EE1[:, k] = la.svd(Y, compute_uv=False)
                        else:
                            U, S, V = la.svd(Y, full_matrices=False)
                            U, S, V = interchsvd(U, oldU, S, V, Nc, k)
                            oldU = U
                            EE1[:, k] = np.diag(S)
                plt.figure(8)
                h2 = plt.plot(s_pass / (2 * np.pi * 1j), EE0.T, 'b-')
                plt.hold(True)
                h3 = plt.plot(s_pass / (2 * np.pi * 1j), EE1.T, 'r--')
                if xlimflag:
                    plt.xlim(opts['plot']['xlim'])
                else:
                    plt.xlim([s_pass[0] / (2 * np.pi * 1j), s_pass[-1] / (2 * np.pi * 1j)])
                if ylimflag:
                    plt.ylim(opts['plot']['ylim'])
                plt.title('Monitoring enforcement process')
                plt.xlabel('Frequency [Hz]')
                plt.hold(False)
                plt.grid(True)
                plt.legend([h2[0], h3[0]], ['Previous', 'Perturbed'])
                plt.draw()
                plt.pause(0.01)
                plt.hold(False)

            tdumtotal = time()
            if iter_in != Niter_in + 1:
                if opts['parametertype'] == 'Y':
                    tdum = time()
                    wintervals = pass_check_Y(SERflag, SER1['poles'], [], SER1['R'], SER1['D'])
                    t[0] += time() - tdum
                    tdum = time()
                    s_viol = violextremaY(SERflag, wintervals.T, SER1['poles'], [], SER1['R'], SER1['D'], colinterch)
                    t[1] += time() - tdum
                elif opts['parametertype'] == 'S':
                    tdum = time()
                    wintervals = pass_check_S(SERflag, SER1['poles'], [], SER1['R'], SER1['D'], TOL, spy, colinterch)
                    t[0] += time() - tdum
                    tdum = time()
                    s_viol = violextremaS(SERflag, wintervals.T, SER1['poles'], [], SER1['R'], SER1['D'], colinterch)
                    t[1] += time() - tdum
                olds3 = s3
                s3 = np.concatenate([s3, s2, s_viol.T])

            if iter_in == Niter_in + 1:
                s3 = []
                s2 = []
                if plotte == 1:
                    EE0 = EE1
                SER0 = SER1

            t[3] += time() - tdumtotal

    if plotte == 1:
        if opts['parametertype'] == 'Y':
            oldT0 = []
            tell = 0
            for k in range(len(s_pass)):
                Y = SER1['C'] @ np.diag((s_pass[k] * I - np.diag(SER1['A'])) ** (-1)) @ SER1['B'] + SER1['D'] + s_pass[k] * SER1['E']
                G = np.real(Y)
                T0, D = la.eig(G)
                T0 = rot(T0)  # minimizing phase angle of eigenvectors in least squares sense
                T0, D = intercheig(T0, oldT0, D, Nc, k)
                oldT0 = T0
                EE1[:, k] = np.diag(D)
        elif opts['parametertype'] == 'S':
            oldU = []
            tell = 0
            for k in range(len(s_pass)):
                Y = SER1['C'] @ np.diag((s_pass[k] * I - np.diag(SER1['A'])) ** (-1)) @ SER1['B'] + SER1['D'] + s_pass[k] * SER1['E']
                if colinterch == 0:
                    EE1[:, k] = la.svd(Y, compute_uv=False)
                else:
                    U, S, V = la.svd(Y, full_matrices=False)
                    U, S, V = interchsvd(U, oldU, S, V, Nc, k)
                    oldU = U
                    EE1[:, k] = np.diag(S)

        plt.figure(7)
        h1 = plt.plot(s_pass / (2 * np.pi * 1j), EE1.T, 'r--')
        if xlimflag:
            plt.xlim(opts['plot']['xlim'])
        else:
            plt.xlim([s_pass[0] / (2 * np.pi * 1j), s_pass[-1] / (2 * np.pi * 1j)])
        if ylimflag:
            plt.ylim(opts['plot']['ylim'])
        plt.legend([h0[0], h1[0]], ['Original', 'Perturbed'], 1)
        plt.hold(False)
        plt.gcf().set_paperunits('centimeters')
        newpos = [0.25, 2.5, 9.5, 9]
        plt.gcf().set_paperpos(newpos)

    if len(wintervals) == 0:
        if outputlevel == 1:
            print(' ')
        print('-->Passivity was successfully enforced.')
        if outputlevel == 1:
            if opts['parametertype'] == 'Y':
                print('   Max. violation, eig(G) :  None')
                print('   Max. violation, eig(D) :  None')
                print('   Max. violation, eig(E) :  None')
            elif opts['parametertype'] == 'S':
                print('   Max. violation, sing(S) :  None')
                print('   Max. violation, sing(D) :  None')
    else:
        print(f'   ***Max. violation, eig(G) : {np.min(g_pass)}')
        print(f'   ***Max. violation, eig(D) : {np.min(la.eig(SER0["D"]))}')
        print(f'   ***Max. violation, eig(E) : {np.min(la.eig(SER0["E"]))}')
        print('-->Iterations terminated before completing passivity enforcement.')
        print('   Increase parameter opts.Niter_out.')

    Ns = len(np.squeeze(s))
    bigYfit = np.zeros((Nc, Nc, Ns), dtype=complex)
    I = np.ones((SER['A'][:, 0].shape[0], 1))
    for k in range(Ns):
        tmp = 1 / (s[:, k] * I - SER1['A'].diagonal().reshape(-1, 1))
        # Y = SER1['C'] @ np.diag((s[k] * I - np.diag(SER1['A'])) ** (-1)) @ SER1['B'] + SER1['D'] + s[k] * SER1['E']
        Y = SER1['C'] @ np.diag(tmp.flatten()) @ SER1['B'] + SER1['D'] + s[:, k] * SER1['E']
        bigYfit[:, :, k] = Y

    if opts['cmplx_ss'] == 0:
        N = len(SER1['A'])
        cindex = np.zeros(N)
        for m in range(N):
            if np.imag(SER1['A'][m, m]) != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        cindex[m + 1] = 2
                    else:
                        cindex[m] = 2
        n = 0
        for m in range(N):
            n += 1
            if cindex[m] == 1:
                a = SER1['A'][n, n]
                a1 = np.real(a)
                a2 = np.imag(a)
                c = SER1['C'][:, n]
                c1 = np.real(c)
                c2 = np.imag(c)
                b = SER1['B'][n, :]
                b1 = 2 * np.real(b)
                b2 = -2 * np.imag(b)
                Ablock = np.array([[a1, a2], [-a2, a1]])

                SER1['A'][n:n + 1, n:n + 1] = Ablock
                SER1['C'][:, n] = c1
                SER1['C'][:, n + 1] = c2
                SER1['B'][n, :] = b1
                SER1['B'][n + 1, :] = b2

    dum = t[0] + t[1]
    if outputlevel == 1:
        print('Time summary: ')
        print(f'   Passivity assessment : {dum} sec')
        print(f'   Passivity enforcement: {t[2]} sec')
        print(f'   Total: {t[3]} sec')
    print('-------------------E N D----------------------------')

    return SER1, bigYfit, opts2

import numpy as np
from scipy.linalg import block_diag, eig, svd

def pass_check_Y(SERflag, A, B, C, D, colinterch):
    wintervals = []

    if SERflag == 1:  # Must convert from pole-residue to state-space
        Nc = len(D)
        N = len(A)
        tell = 0
        CC = np.zeros((Nc, Nc * N))
        AA = np.array([])
        BB = np.array([])
        B = np.ones((N, 1))
        for col in range(Nc):
            AA = block_diag(np.diag(A))  # AA = block_diag(AA, np.diag(A))，matlab和python中此函数不一样，当AA是空数组
            BB = block_diag(B)  # BB = block_diag(BB, B)
            for row in range(col, Nc):
                CC[row, col * N:(col + 1) * N] = C[row, col, :]
                CC[col, row * N:(row + 1) * N] = C[row, col, :]
        A = AA
        B = BB
        C = CC

    Acmplx = A
    Bcmplx = B
    Ccmplx = C
    Dcmplx = D

    if np.sum(A - np.diag(np.diag(A))) == 0:  # Convert to real-only
        N = len(A)
        cindex = np.zeros(N)
        for m in range(N):
            if np.imag(A[m, m]) != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        cindex[m + 1] = 2
                    else:
                        cindex[m] = 2

        n = 0
        for m in range(N):
            n += 1
            if cindex[m] == 1:
                a = A[n, n]
                a1 = np.real(a)
                a2 = np.imag(a)
                c = C[:, n]
                c1 = np.real(c)
                c2 = np.imag(c)
                b = B[n, :]
                b1 = 2 * np.real(b)
                b2 = -2 * np.imag(b)
                Ablock = np.array([[a1, a2], [-a2, a1]])
                A[n:n + 1, n:n + 1] = Ablock
                C[:, n] = c1
                C[:, n + 1] = c2
                B[n, :] = b1
                B[n + 1, :] = b2

    N = len(A)
    Nc = len(D)
    tell = 0

    E = np.zeros((Nc, Nc))  # Dummy E-matrix
    D = D.reshape(1, -1)

    D_eigenvalues = eig(D)[0]
    if np.sum(D_eigenvalues == 0) > 0:  # singular D;
        # 计算 A 的逆
        Ahat = np.linalg.inv(A)

        # 计算 Bhat, Chat, Dhat
        Bhat = -np.dot(Ahat, B)
        Chat = np.dot(C, Ahat)
        Dhat = D - np.dot(C, np.dot(Ahat, B))

        # 更新 A, B, C, D
        A = Ahat
        B = Bhat
        C = Chat
        D = Dhat

    S1 = A @ (B @ np.linalg.inv(D) @ C - A)  # 计算 S1 矩阵
    wS1 = eig(S1)[0]  # 计算 S1 的特征值
    wS1 = np.sqrt(wS1)  # 对特征值取平方根

    Dcmplx = Dcmplx.reshape(1, -1)
    Dcmplx_eigenvalues = eig(Dcmplx)[0]
    if np.sum(Dcmplx_eigenvalues == 0) > 0:
        wS1 = 1.0 / wS1

    ind = np.where(np.imag(wS1) == 0)[0]
    wS1 = wS1[ind]
    sing_w = np.sort(wS1)

    if len(sing_w) == 0:
        sing_w = []
        intervals = []
        return wintervals

    A = Acmplx
    B = Bcmplx
    C = Ccmplx
    D = Dcmplx

    midw = np.zeros(1 + len(sing_w))
    midw[0] = sing_w[0] / 2
    midw[-1] = 2 * sing_w[-1]
    for k in range(len(sing_w) - 1):
        midw[k + 1] = (sing_w[k] + sing_w[k + 1]) / 2

    # Checking passivity at all midpoints:
    for k in range(len(midw)):
        sk = 1j * midw[k]
        G = np.real(fitcalcABCDE(sk, np.diag(A), B, C, D, E))
        EE[:, k] = eig(G)
        if np.any(EE[:, k] < 0):
            viol[k] = 1
        else:
            viol[k] = 0

    # Establishing intervals for passivity violations:
    intervals = []
    for k in range(len(midw)):
        if viol[k] == 1:
            if k == 0:
                intervals = np.append(intervals, [[0, sing_w[0]]], axis=1)  # The first violations starts at DC
            elif k == len(midw) - 1:
                intervals = np.append(intervals, [[sing_w[k - 1], 1e16]], axis=1)  # The last violation extends to infinite frequency
            else:
                intervals = np.append(intervals, [[sing_w[k - 1], sing_w[k]]], axis=1)

    if len(intervals) == 0:
        wintervals = intervals
        return

    # Collapsing overlapping bands:
    tell = 0
    killindex = 0
    for k in range(1, len(intervals[0, :])):
        if intervals[1, k - 1] == intervals[0, k]:  # An overlap exists
            tell += 1
            intervals[1, k - 1] = intervals[1, k]  # Extending interval
            intervals[:, k] = intervals[:, k - 1]  # Copying interval
            killindex[tell] = k - 1

    if killindex != 0:
        intervals = np.delete(intervals, killindex, axis=1)

    wintervals = intervals
    return wintervals

def pass_check_S(SERflag, A, B, C, D, TOL, spy, colinterch):
    wintervals = []

    if SERflag == 1:  # Must convert from pole-residue to state-space
        Nc = len(D)
        N = len(A)
        tell = 0
        CC = np.zeros((Nc, Nc * N))
        AA = np.array([])
        BB = np.array([])
        B = np.ones((N, 1))
        for col in range(Nc):
            AA = block_diag(AA, np.diag(A))
            BB = block_diag(BB, B)
            for row in range(col, Nc):
                CC[row, (col - 1) * N:col * N] = C[row, col, :]
                CC[col, (row - 1) * N:row * N] = C[row, col, :]
        A = AA
        B = BB
        C = CC

    Acmplx = A
    Bcmplx = B
    Ccmplx = C

    if np.sum(A - np.diag(np.diag(A))) == 0:  # Convert to real-only
        N = len(A)
        cindex = np.zeros(N)
        for m in range(N):
            if np.imag(A[m, m]) != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        cindex[m + 1] = 2
                    else:
                        cindex[m] = 2

        n = 0
        for m in range(N):
            n += 1
            if cindex[m] == 1:
                a = A[n, n]
                a1 = np.real(a)
                a2 = np.imag(a)
                c = C[:, n]
                c1 = np.real(c)
                c2 = np.imag(c)
                b = B[n, :]
                b1 = 2 * np.real(b)
                b2 = -2 * np.imag(b)
                Ablock = np.array([[a1, a2], [-a2, a1]])
                A[n:n + 1, n:n + 1] = Ablock
                C[:, n] = c1
                C[:, n + 1] = c2
                B[n, :] = b1
                B[n + 1, :] = b2

    N = len(A)
    Nc = len(D)
    tell = 0

    # Calculating Hamiltonian matrix:
    I = np.eye(len(D))
    R = D.T @ D - I
    invR = np.linalg.inv(R)

    P = (A - B @ np.linalg.inv(D - I) @ C) @ (A - B @ np.linalg.inv(D + I) @ C)
    wS1 = np.sqrt(eig(P))
    ind = np.where(np.real(wS1) == 0)[0]
    wS1 = np.imag(wS1[ind])
    singulars = 1j * np.sort(wS1)
    A = Acmplx
    B = Bcmplx
    C = Ccmplx
    E = np.zeros((Nc, Nc))  # Dummy E-matrix

    if len(singulars) == 0:
        sing_w = []
        intervals = []
        return

    sing_w = np.imag(singulars)  # Crossing frequencies [rad/sec]
    sing_w = np.sort(sing_w)

    # Establising frequency list at midpoint of all bands defeined by sing_w:
    midw = np.zeros(1 + len(sing_w))
    midw[0] = sing_w[0] / 2
    midw[-1] = 2 * sing_w[-1]
    for k in range(len(sing_w) - 1):
        midw[k + 1] = (sing_w[k] + sing_w[k + 1]) / 2

    # Checking passivity at all midpoints:
    oldU = []
    for k in range(len(midw)):
        sk = 1j * midw[k]
        Y = fitcalcABCDE(sk, np.diag(A), B, C, D, E)
        if colinterch == 0:
            EE[:, k] = svd(Y, 0)
        else:
            U, S, V = svd(Y, 0)
            U, S, V = interchsvd(U, oldU, S, V, Nc, k)
            oldU = U
            EE[:, k] = np.diag(S)
        if np.any(EE[:, k] > 1):
            viol[k] = 1
        else:
            viol[k] = 0

    # Establishing intervals for passivity violations:
    intervals = []
    for k in range(len(midw)):
        if viol[k] == 1:
            if k == 0:
                intervals = np.append(intervals, [[0, sing_w[0]]], axis=1)  # The first violations starts at DC
            elif k == len(midw) - 1:
                intervals = np.append(intervals, [[sing_w[k - 1], 1e16]], axis=1)  # The last violation extends to infinite frequency
            else:
                intervals = np.append(intervals, [[sing_w[k - 1], sing_w[k]]], axis=1)

    if len(intervals) == 0:
        wintervals = intervals
        return

    # Collapsing overlapping bands:
    tell = 0
    killindex = 0
    for k in range(1, len(intervals[0, :])):
        if intervals[1, k - 1] == intervals[0, k]:  # An overlap exists
            tell += 1
            intervals[1, k - 1] = intervals[1, k]  # Extending interval
            intervals[:, k] = intervals[:, k - 1]  # Copying interval
            killindex[tell] = k - 1

    if killindex != 0:
        intervals = np.delete(intervals, killindex, axis=1)

    wintervals = intervals
    return wintervals

