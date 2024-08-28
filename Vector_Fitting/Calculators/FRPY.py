import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize
from Vector_Fitting.Calculators.pr2ss import pr2ss
from Calculators.fitcalcPRE import fitcalcPRE

# 暂时注释掉了49行的clear bigA
def FRPY(SER, s, s2, s3, RPopts):
    auxflag = RPopts['auxflag']
    weightfactor = RPopts['weightfactor']
    weightparam = RPopts['weightparam']
    if 'weight' in RPopts:
        bigweight = RPopts['weight']
    TOLE = RPopts['TOLE']
    TOL = RPopts['TOLGD']

    SERA = SER['poles']
    if SERA.shape[0] < SERA.shape[1]:
        SERA = SERA.T  # Ensure column vector
    SERC = SER['R']
    SERD = SER['D']
    SERE = SER['E']

    if 'H' not in RPopts:
        RPopts['H'] = []
        RPopts['oldDflag'] = -1
        RPopts['oldEflag'] = -1

    d = np.linalg.eigvals(SERD)
    eigD = d
    if np.any(d < 0):
        Dflag = 1  # Will perturb D-matrix
        VD, eigD = np.linalg.eig(SERD)
        invVD = np.linalg.inv(VD)
        eigD = np.diag(eigD)
    else:
        Dflag = 0

    e = np.linalg.eigvals(SERE)
    eigE = e
    if np.any(e < 0):
        Eflag = 1  # Will perturb E-matrix
        VE, eigE = np.linalg.eig(SERE)
        invVE = np.linalg.inv(VE)
        eigE = np.diag(eigE)
    else:
        Eflag = 0

    SERCnew = SERC
    SERDnew = SERD
    SEREnew = SERE

    N = len(SERA)
    bigB = []
    bigc = []
    Ns = len(s)
    Ns2 = len(s2)
    Nc = len(SERD)
    Nc2 = Nc * Nc
    I = np.eye(Nc)
    M2mat = []

    # Finding out which poles are complex:
    cindex = np.zeros(N)
    for m in range(N):
        if np.imag(SERA[m]) != 0:
            if m == 0:
                cindex[m] = 1
            else:
                if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                    cindex[m] = 1
                    cindex[m + 1] = 2
                else:
                    cindex[m] = 2

    if not RPopts['H']:
        if RPopts['outputlevel'] == 1:
            print('    Building system equation (once)...')

        if (Dflag + Eflag) == 2:
            bigA = np.zeros((Ns * Nc2, Nc * (N + 2)))
        elif (Dflag + Eflag) == 1:
            bigA = np.zeros((Ns * Nc2, Nc * (N + 1)))
        else:
            bigA = np.zeros((Ns * Nc2, Nc * N))

        for m in range(N):
            R = SERC[:, :, m]
            if cindex[m] == 0:  # real pole
                R = R
            elif cindex[m] == 1:  # complex pole, 1st part
                R = np.real(R)
            else:
                R = np.imag(R)
            V, D = np.linalg.eig(R)
            bigV[:, (m - 1) * Nc:m * Nc] = V
            biginvV[:, (m - 1) * Nc:m * Nc] = np.linalg.inv(V)
            bigD[:, m] = np.diag(D)

        for k in range(Ns):
            sk = s[k]
            # Calculating matrix Mmat (coefficients for LS-problem)
            tell = 0
            offs = 0
            Yfit = fitcalcPRE(sk, SERA, SERC, SERD, SERE)

            if weightparam is not None:
                if weightparam == 1:
                    weight = np.ones((Nc, Nc))
                elif weightparam == 2:
                    weight = 1.0 / np.abs(Yfit)
                elif weightparam == 3:
                    weight = 1.0 / np.sqrt(np.abs(Yfit))
                elif weightparam == 4:
                    weight = np.ones((Nc, Nc)) / np.linalg.norm(np.abs(Yfit))
                elif weightparam == 5:
                    weight = np.ones((Nc, Nc)) / np.sqrt(np.linalg.norm(np.abs(Yfit)))
                else:
                    weight = 1
            else:
                weight = bigweight[:, :, k]

            for m in range(N):
                V = bigV[:, (m - 1) * Nc:m * Nc]
                invV = np.linalg.inv(V)
                if cindex[m] == 0:  # real pole
                    dum = 1 / (sk - SERA[m])
                elif cindex[m] == 1:  # complex pole, 1st part
                    dum = (1 / (sk - SERA[m]) + 1 / (sk - SERA[m].conj()))
                else:
                    dum = (1j / (sk - SERA[m].conj()) - 1j / (sk - SERA[m]))

                for egenverdi in range(Nc):
                    tell = 0
                    gamm = V[:, egenverdi] @ invV[egenverdi, :]
                    for row in range(Nc):
                        for col in range(Nc):
                            faktor = weight[row, col]
                            tell += 1
                            if cindex[m] == 0:  # real pole
                                Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor * dum
                            elif cindex[m] == 1:  # complex pole, 1st part
                                Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor * dum
                            else:
                                Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor * dum
                offs += Nc

            if Dflag == 1:
                for egenverdi in range(Nc):  # Eigenvalues for residue matrix
                    gamm = VD[:, egenverdi] @ invVD[egenverdi, :]  # Outer product
                    tell = 0
                    for row in range(Nc):
                        for col in range(Nc):
                            tell += 1
                            faktor = weight[row, col]
                            Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor

            if Eflag == 1:
                for egenverdi in range(Nc):  # Eigenvalues for residue matrix
                    gamm = VE[:, egenverdi] @ invVE[egenverdi, :]  # Outer product
                    tell = 0
                    for row in range(Nc):
                        for col in range(Nc):
                            tell += 1
                            faktor = weight[row, col]
                            Mmat[tell, offs + Nc * Dflag + egenverdi] = gamm[row, col] * sk * faktor

            bigA[(k - 1) * Nc2 + 1:k * Nc2, :] = Mmat

        # INTRODUCING SAMPLES OUTSIDE LS REGION: ONE SAMPLE PER POLE (s4)
        if auxflag == 1:
            s4 = []
            tell = 0
            for m in range(len(SERA)):
                if cindex[m] == 0:  # real pole
                    if (np.abs(SERA[m]) > s[Ns - 1] / 1j) or (np.abs(SERA[m]) < s[0] / 1j):
                        tell += 1
                        s4.append(1j * np.abs(SERA[m]))
                elif cindex[m] == 1:  # complex pole, first part
                    if (np.abs(np.imag(SERA[m])) > s[Ns - 1] / 1j) or (np.abs(np.imag(SERA[m])) < s[0] / 1j):
                        tell += 1
                        s4.append(1j * np.abs(np.imag(SERA[m])))

            Ns4 = len(s4)
            bigA2 = np.zeros((Ns4 * Nc2, Nc * (N + Dflag + Eflag)))

            for k in range(Ns4):
                sk = s4[k]
                # Calculating matrix Mmat (coefficients for LS-problem)
                tell = 0
                offs = 0
                Yfit = fitcalcPRE(sk, SERA, SERC, SERD, SERE)

                if weightparam == 1:
                    weight = np.ones((Nc, Nc))
                elif weightparam == 2:
                    weight = 1.0 / np.abs(Yfit)
                elif weightparam == 3:
                    weight = 1.0 / np.sqrt(np.abs(Yfit))
                elif weightparam == 4:
                    weight = np.ones((Nc, Nc)) / np.linalg.norm(np.abs(Yfit))
                elif weightparam == 5:
                    weight = np.ones((Nc, Nc)) / np.sqrt(np.linalg.norm(np.abs(Yfit)))
                else:
                    weight = 1

                weight = weight * weightfactor

                for m in range(N):
                    V = bigV[:, (m - 1) * Nc:m * Nc]
                    invV = np.linalg.inv(V)
                    if cindex[m] == 0:  # real pole
                        dum = 1 / (sk - SERA[m])
                    elif cindex[m] == 1:  # complex pole, 1st part
                        dum = (1 / (sk - SERA[m]) + 1 / (sk - SERA[m].conj()))
                    else:
                        dum = (1j / (sk - SERA[m].conj()) - 1j / (sk - SERA[m]))

                    for egenverdi in range(Nc):
                        tell = 0
                        gamm = V[:, egenverdi] @ invV[egenverdi, :]
                        for row in range(Nc):
                            for col in range(Nc):
                                faktor = weight[row, col]
                                tell += 1
                                if cindex[m] == 0:  # real pole
                                    Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor * dum
                                elif cindex[m] == 1:  # complex pole, 1st part
                                    Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor * dum
                                else:
                                    Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor * dum
                    offs += Nc

                if Dflag == 1:
                    for egenverdi in range(Nc):  # Eigenvalues for residue matrix
                        gamm = VD[:, egenverdi] @ invVD[egenverdi, :]  # Outer product
                        tell = 0
                        for row in range(Nc):
                            for col in range(Nc):
                                tell += 1
                                faktor = weight[row, col]
                                Mmat[tell, offs + egenverdi] = gamm[row, col] * faktor

                if Eflag == 1:
                    for egenverdi in range(Nc):  # Eigenvalues for residue matrix
                        gamm = VE[:, egenverdi] @ invVE[egenverdi, :]  # Outer product
                        tell = 0
                        for row in range(Nc):
                            for col in range(Nc):
                                tell += 1
                                faktor = weight[row, col]
                                Mmat[tell, offs + Nc * Dflag + egenverdi] = gamm[row, col] * sk * faktor

                bigA2[(k - 1) * Nc2 + 1:k * Nc2, :] = Mmat

            bigA = np.vstack((bigA, bigA2))

        bigA = np.vstack((np.real(bigA), np.imag(bigA)))
        Acol = bigA.shape[1]
        for col in range(Acol):
            Escale[col] = np.linalg.norm(bigA[:, col], 2)
            bigA[:, col] /= Escale[col]

        H = bigA.T @ bigA
        RPopts['H'] = H
        RPopts['Escale'] = Escale
        RPopts['bigV'] = bigV
        RPopts['biginvV'] = biginvV
        if RPopts['outputlevel'] == 1:
            print('    Done')

    else:
        bigV = RPopts['bigV']
        biginvV = RPopts['biginvV']
        if Dflag != RPopts['oldDflag'] or Eflag != RPopts['oldEflag']:
            RPopts['H'] = RPopts['H'][:Nc * (N + Dflag + Eflag), :Nc * (N + Dflag + Eflag)]
            RPopts['Escale'] = RPopts['Escale'][:Nc * (N + Dflag + Eflag)]

    Mmat2 = np.zeros((Nc2, Nc * (N + Dflag + Eflag)))
    viol_G = []
    viol_D = []
    viol_E = []

    # LOOP FOR CONSTRAINT PROBLEM, TYPE #1 (violating eigenvalues in s2):
    for k in range(Ns2):
        sk = s2[k]
        for row in range(Nc):
            for col in range(Nc):
                Y[row, col] = SERD[row, col] + sk * SERE[row, col]
                Y[row, col] += np.sum(SERC[row, col, :] / (sk - SERA))

        # Calculating eigenvalues and eigenvectors:
        V, Z = np.linalg.eig(np.real(Y))
        Z = np.diag(Z)
        EE[:, k] = np.real(Z)
        if np.min(np.real(Z)) < 0:  # any violations

            # Calculating matrix M2mat; matrix of partial derivatives:
            tell = 0
            offs = 0

            for m in range(N):
                VV = bigV[:, (m - 1) * Nc:m * Nc]
                invVV = biginvV[:, (m - 1) * Nc:m * Nc]
                for egenverdi in range(Nc):
                    tell = 0
                    gamm = VV[:, egenverdi] @ invVV[egenverdi, :]
                    for row in range(Nc):
                        for col in range(Nc):
                            tell += 1
                            if cindex[m] == 0:  # real pole
                                Mmat2[tell, offs + egenverdi] = gamm[row, col] / (sk - SERA[m])
                            elif cindex[m] == 1:  # complex pole, 1st part
                                Mmat2[tell, offs + egenverdi] = gamm[row, col] * (1 / (sk - SERA[m]) + 1 / (sk - SERA[m].conj()))
                            else:
                                Mmat2[tell, offs + egenverdi] = gamm[row, col] * (1j / (sk - SERA[m].conj()) - 1j / (sk - SERA[m]))
                offs += Nc

            if Dflag == 1:
                for egenverdi in range(Nc):  # Eigenvalues for residue matrix
                    tell = 0
                    gamm = VD[:, egenverdi] @ invVD[egenverdi, :]  # Outer product
                    for row in range(Nc):
                        for col in range(Nc):
                            tell += 1
                            Mmat2[tell, offs + egenverdi] = gamm[row, col]

            for n in range(Nc):
                tell = 0
                V1 = V[:, n]
                for row in range(Nc):
                    for col in range(Nc):
                        if row == col:
                            qij = V1[row] ** 2
                        else:
                            qij = V1[row] * V1[col]
                        tell += 1
                        Q[n, tell] = qij

            B = Q @ Mmat2
            delz = np.real(Z)
            for n in range(Nc):  # instability?
                if delz[n] < 0:
                    bigB = np.vstack((bigB, B[n, :]))
                    bigc = np.vstack((bigc, -TOL + delz[n]))
                    viol_G = np.vstack((viol_G, delz[n]))

    # LOOP FOR CONSTRAINT PROBLEM, TYPE #2: (all eigenvalues in s3):
    Ns3 = len(s3)
    for k in range(Ns3):
        sk = s3[k]
        for row in range(Nc):
            for col in range(Nc):
                Y[row, col] = SERD[row, col] + sk * SERE[row, col]
                Y[row, col] += np.sum(SERC[row, col, :] / (sk - SERA))

        # Calculating eigenvalues and eigenvectors:
        V, Z = np.linalg.eig(np.real(Y))
        Z = np.diag(Z)
        EE[:, k] = np.real(Z)

        # Calculating matrix M2mat; matrix of partial derivatives:
        tell = 0
        offs = 0

        for m in range(N):
            VV = bigV[:, (m - 1) * Nc:m * Nc]
            invVV = biginvV[:, (m - 1) * Nc:m * Nc]
            for egenverdi in range(Nc):
                tell = 0
                gamm = VV[:, egenverdi] @ invVV[egenverdi, :]
                for row in range(Nc):
                    for col in range(Nc):
                        tell += 1
                        if cindex[m] == 0:  # real pole
                            Mmat2[tell, offs + egenverdi] = gamm[row, col] / (sk - SERA[m])
                        elif cindex[m] == 1:  # complex pole, 1st part
                            Mmat2[tell, offs + egenverdi] = gamm[row, col] * (1 / (sk - SERA[m]) + 1 / (sk - SERA[m].conj()))
                        else:
                            Mmat2[tell, offs + egenverdi] = gamm[row, col] * (1j / (sk - SERA[m].conj()) - 1j / (sk - SERA[m]))
            offs += Nc

        if Dflag == 1:
            for egenverdi in range(Nc):  # Eigenvalues for residue matrix
                tell = 0
                gamm = VD[:, egenverdi] @ invVD[egenverdi, :]  # Outer product
                for row in range(Nc):
                    for col in range(Nc):
                        tell += 1
                        Mmat2[tell, offs + egenverdi] = gamm[row, col]

        for n in range(Nc):
            tell = 0
            V1 = V[:, n]
            for row in range(Nc):
                for col in range(Nc):
                    if row == col:
                        qij = V1[row] ** 2
                    else:
                        qij = V1[row] * V1[col]
                    tell += 1
                    Q[n, tell] = qij

        B = Q @ Mmat2
        delz = np.real(Z)
        for n in range(Nc):
            bigB = np.vstack((bigB, B[n, :]))
            bigc = np.vstack((bigc, -TOL + delz[n]))
            viol_G = np.vstack((viol_G, delz[n]))

    # Adding constraint for possible eigenvalues in D < 0
    if Dflag == 1:
        for n in range(Nc):
            dum = np.zeros((1, (Nc * (N + Dflag + Eflag))))
            dum[Nc * N + n] = 1
            bigB = np.vstack((bigB, dum))
            bigc = np.vstack((bigc, eigD[n] - TOL))
            viol_G = np.vstack((viol_G, eigD[n]))
            viol_D = np.vstack((viol_D, eigD[n]))

    # Adding constraint for possible eigenvalues in E < 0
    if Eflag == 1:
        for n in range(Nc):
            dum = np.zeros((1, (Nc * (N + Dflag + Eflag))))
            dum[Nc * (N + Dflag) + n] = 1
            bigB = np.vstack((bigB, dum))
            bigc = np.vstack((bigc, eigE[n] - TOLE))
            viol_E = np.vstack((viol_E, eigE[n]))

    if len(bigB) == 0:
        return  # No passivity violations

    c = bigc
    bigB = np.real(bigB)
    for col in range(len(RPopts['H'])):
        if len(bigB) > 0:
            bigB[:, col] /= RPopts['Escale'][col]

    ff = np.zeros((len(RPopts['H']), 1))
    # clear bigA

    if RPopts['solver'] == 'QUADPROG':
        dx = minimize(lambda x: 0.5 * x.T @ RPopts['H'] @ x + ff.T @ x, np.zeros(len(RPopts['H'])),
                      constraints={'type': 'ineq', 'fun': lambda x: bigB @ x + bigc})['x']
    elif RPopts['solver'] == 'CPLEX':
        c = 0 * ff
        H0 = c.T
        A = -bigB
        x_0 = np.zeros(len(H[:, 1]))
        b_U = bigc
        b_L = -np.inf * np.ones(len(b_U))
        x_L = -np.inf * np.ones(len(x_0))
        x_U = np.inf * np.ones(len(x_0))
        Prob = qpAssign(H, c, A, b_L, b_U, x_L, x_U, x_0, 'dust')
        Prob.MIP.cpxControl.QPMETHOD = 4
        Prob.MIP.cpxControl.BARALG = 2
        PriLev = 0
        Prob.PriLevOpt = 0
        x, slack, v, rc, f_k = cplex(c, A, x_L, x_U, b_L, b_U, Prob.MIP.cpxControl, [], PriLev, Prob, [], [], [], [], [], [], H)
        dx = x

    dx /= RPopts['Escale'].T

    # Updating eigenvalues and SERC:
    for m in range(N):
        if cindex[m] == 0:  # real pole
            D1 = np.diag(dx[(m - 1) * Nc + 1:m * Nc])
            SERCnew[:, :, m] = SERCnew[:, :, m] + bigV[:, (m - 1) * Nc:m * Nc] @ D1 @ biginvV[:, (m - 1) * Nc:m * Nc]
        elif cindex[m] == 1:  # complex pole, 1st part
            GAMM1 = bigV[:, (m - 1) * Nc:m * Nc]
            GAMM2 = bigV[:, (m + 1 - 1) * Nc:(m + 1) * Nc]
            invGAMM1 = biginvV[:, (m - 1) * Nc:m * Nc]
            invGAMM2 = biginvV[:, (m + 1 - 1) * Nc:(m + 1) * Nc]
            D1 = np.diag(dx[(m - 1) * Nc + 1:m * Nc])
            D2 = np.diag(dx[(m + 1 - 1) * Nc:(m + 1) * Nc])
            R1 = np.real(SERC[:, :, m])
            R2 = np.imag(SERC[:, :, m])
            R1new = R1 + GAMM1 @ D1 @ invGAMM1
            R2new = R2 + GAMM2 @ D2 @ invGAMM2
            SERCnew[:, :, m] = R1new + 1j * R2new
            SERCnew[:, :, m + 1] = R1new - 1j * R2new

    if Dflag == 1:
        DD = np.diag(dx[N * Nc + 1:(N + 1) * Nc])
        SERDnew = SERDnew + VD @ DD @ invVD

    if Eflag == 1:
        EE = np.diag(dx[(N + Dflag) * Nc + 1:(N + Dflag + Eflag) * Nc])
        SEREnew = SEREnew + VE @ EE @ invVE

    # Ensuring symmetry
    SERDnew = (SERDnew + SERDnew.T) / 2
    SEREnew = (SEREnew + SEREnew.T) / 2
    for m in range(N):
        SERCnew[:, :, m] = (SERCnew[:, :, m] + SERCnew[:, :, m].T) / 2

    SER['R'] = SERCnew
    SER['D'] = SERDnew
    SER['E'] = SEREnew
    SER = pr2ss(SER)

    RPopts['oldDflag'] = Dflag
    RPopts['oldEflag'] = Eflag

    return SER, RPopts

