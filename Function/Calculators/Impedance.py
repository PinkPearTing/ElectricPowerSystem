import sys

sys.path.append('../..')
import numpy as np
from scipy.special import iv as besseli
from scipy.special import ivp as besselip
from scipy.special import kv as besselk
from scipy.special import kvp as besselkp
from Utils.Math import Bessel_IK, Bessel_K2

frq_default = np.logspace(0, 9, 37)

def calculate_coreWires_impedance(core_wires_r, core_wires_offset, core_wires_angle, core_wires_mur,
                                  core_wires_sig, core_wires_epr, sheath_mur, sheath_sig, sheath_epr, sheath_inner_radius, Frq, constants):
    """
    【函数功能】芯线阻抗计算
    【入参】
    core_wires_r (numpy.ndarray, n*1): n条芯线的半径
    core_wires_offset (numpy.ndarray, n*1): n条芯线距离中心位置
    core_wires_angle (numpy.ndarray, n*1): n条芯线角度
    core_wires_mur (numpy.ndarray, n*1): n条芯线的磁导率
    core_wires_sig (numpy.ndarray, n*1): n条芯线的电导率
    sheath_mur (float): 套管的磁导率
    sheath_sig (float): 套管的电导率
    sheath_inner_radius (float): 套管的内径
    Frq(numpy.ndarray,1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zc(numpy.ndarray:1*n*Nf): n条芯线在Nf个频率下的阻抗矩阵
    """
    mu0, ep0 = constants.mu0, constants.ep0
    Besl_Max = 200
    Nbesl = 15
    frq = np.array([Frq]).reshape(-1)
    Npha = core_wires_r.shape[0]
    Nf = frq.size
    omega = 2 * np.pi * frq

    Mu_c = mu0 * core_wires_mur[0]
    gamma_c = np.sqrt(1j * Mu_c * omega * (core_wires_sig[0] + 1j * omega * ep0 * core_wires_epr))
    Rc = core_wires_r * gamma_c
    Zc_diag = 1j * omega * Mu_c / (2 * np.pi * Rc) * besseli(0, Rc) / besseli(1, Rc)
    # Zc_diag = 1 / (2 * np.pi * core_wires_r * core_wires_sig[0]) * gamma_c
    # Rc = core_wires_r * gamma_c
    # kc = 1 / (2 * np.pi * core_wires_r) * (1j * omega * Mu_c / gamma_c)
    # low = np.where(np.real(Rc) <= Besl_Max)
    # Zc_diag[low] = kc[low] * besseli(0, Rc[low]) / besseli(1, Rc[low])

###################################
    # Mu_s = mu0 * sheath_mur
    # gamma_s = np.sqrt(1j * Mu_s * omega * (sheath_sig + 1j * omega * ep0* sheath_epr))
    # Rsa = sheath_inner_radius * gamma_s
    # tmat = np.tile(core_wires_angle, (1, Npha))
    # angle = (tmat - tmat.T) * np.pi / 180
    # didk = core_wires_offset * core_wires_offset.T
    # ks = 1j * omega * mu0 / (2 * np.pi)
    #
    # dj = np.tile(core_wires_offset, (1, Npha))
    # jwL = ks * np.log(dj.T/sheath_inner_radius*np.sqrt((didk**2+sheath_inner_radius**4-2*didk*sheath_inner_radius**2*np.cos(angle))/(didk**2+dj.T**4-2*didk*dj.T**2*np.cos(angle))))
    # jwL_diag = ks * np.log(sheath_inner_radius/core_wires_r*(1-(core_wires_offset/sheath_inner_radius)**2))
    # np.fill_diagonal(jwL, jwL_diag)
    #
    # temp_Zsi = 0
    # temp_Lcs = 0
    # for ik in range(1, Nbesl+1):
    #     temp_Zsi += sheath_mur * (didk / sheath_inner_radius**2) ** ik * np.cos(ik*angle) * 2 / (ik * sheath_mur - Rsa * besselkp(ik, Rsa, 1) / besselk(ik, Rsa))
    #     temp_Lcs += (didk / sheath_inner_radius**2) ** ik * np.cos(ik*angle)/ik
    # np.fill_diagonal(temp_Lcs, 0)
    #
    # # Zaa = gamma_s / 2 / np.pi / sheath_sig / sheath_inner_radius * (
    # #             besseli(0, Rsa) * besselk(1, Rsb) + besseli(1, Rsb) * besselk(0, Rsa)) / (
    # #                   besseli(1, Rsb) * besselk(1, Rsa) - besseli(1, Rsa) * besselk(1, Rsb))
    #
    # dj = np.tile(core_wires_offset, (1, Npha))
    # tempL = np.log(sheath_inner_radius/np.sqrt(dj**2+dj.T**2-2*didk*np.cos(angle)))
    # tempL_diag = np.log(sheath_inner_radius/core_wires_r*(1-(core_wires_offset/sheath_inner_radius)**2))
    # np.fill_diagonal(tempL, tempL_diag)
    # tempL -= temp_Lcs
    #
    # Nc = core_wires_r.shape[0]
    # Zc = np.zeros((Nc, Nc, Nf), dtype='complex')
    #
    # for ik in range(Nf):
    #     Zc[:, :, ik] = np.diag(Zc_diag[:, ik]) + ks[ik] * (sheath_mur * besselk(0, Rsa) / Rsa / besselk(1, Rsa) + temp_Zsi + tempL)

    return Zc_diag

def calculate_round_wires_internal_impedance(core_wires_r, core_wires_mur, core_wires_sig, core_wires_epr, frq, constants):
    """
    【函数功能】圆线内阻抗计算
    【入参】
    core_wires_r (numpy.ndarray, n*1): n条圆线的半径
    core_wires_mur (numpy.ndarray, n*1): n条圆线的磁导率
    core_wires_sig (numpy.ndarray, n*1): n条圆线的电导率
    core_wires_epr (numpy.ndarray, n*1): n条圆线的介电常数
    Frq(numpy.ndarray,1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zc(numpy.ndarray:n*n*Nf): n条芯线在Nf个频率下的阻抗矩阵
    """
    mu0, ep0 = constants.mu0, constants.ep0
    omega = 2 * np.pi * frq

    Mu_c = mu0 * core_wires_mur[0]
    gamma_c = np.sqrt(1j * Mu_c * omega * (core_wires_sig[0] + 1j * omega * ep0 * core_wires_epr))
    Rc = core_wires_r * gamma_c
    Zc_diag = 1j * omega * Mu_c / (2 * np.pi * Rc) * besseli(0, Rc) / besseli(1, Rc)
    return Zc_diag

def calculate_inductance_of_round_wires_inside_sheath(core_wires_r, core_wires_offset, core_wires_angle, sheath_inner_radius, constants):
    """
    【函数功能】套管内的圆线电感计算
    【入参】
    core_wires_r (numpy.ndarray, n*1): n条圆线的半径
    core_wires_offset (numpy.ndarray, n*1): n条圆线距中心距离
    core_wires_angle (numpy.ndarray, n*1): n条圆线的角度，单位°
    sheath_inner_radius (float): 套管的内径

    【出参】
    L(numpy.ndarray:n*n): n条圆线的电感矩阵
    """
    mu0, ep0 = constants.mu0, constants.ep0
    Npha = core_wires_r.shape[0]
    tmat = np.tile(core_wires_angle, (1, Npha))
    angle = (tmat - tmat.T) * np.pi / 180
    didk = core_wires_offset * core_wires_offset.T
    ks = mu0 / (2 * np.pi)

    dj = np.tile(core_wires_offset, (1, Npha))
    L = ks * np.log(dj.T/sheath_inner_radius*np.sqrt((didk**2+sheath_inner_radius**4-2*didk*sheath_inner_radius**2*np.cos(angle))/(didk**2+dj.T**4-2*didk*dj.T**2*np.cos(angle))))
    L_diag = ks * np.log(sheath_inner_radius/core_wires_r*(1-(core_wires_offset/sheath_inner_radius)**2))
    np.fill_diagonal(L, L_diag)
    return L

def calculate_sheath_internal_impedance(sheath_mur, sheath_sig, sheath_epr, sheath_inner_radius, sheath_r, frq, constants):
    """
    【函数功能】套管内的圆线电感计算
    【入参】
    sheath_mur(float): 套管的磁导率
    sheath_sig(float): 套管的电导率
    sheath_epr(float): 套管的介电常数
    sheath_inner_radius (float): 套管的内径
    sheath_r (float): 套管的半径，不包括绝缘层
    Frq(numpy.ndarray,1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zc(numpy.ndarray:1*Nf): 套管在Nf个频率下的内阻抗矩阵
    """
    mu0, ep0 = constants.mu0, constants.ep0
    omega = 2 * np.pi * frq

    Mu_s = mu0 * sheath_mur
    gamma_s = np.sqrt(1j * Mu_s * omega * (sheath_sig + 1j * omega * ep0* sheath_epr))
    Rsa = sheath_inner_radius * gamma_s
    Rsb = sheath_r * gamma_s

    Zinternal = gamma_s / 2 / np.pi / sheath_sig / sheath_inner_radius * (
                besseli(0, Rsa) * besselk(1, Rsb) + besseli(1, Rsb) * besselk(0, Rsa)) / (
                      besseli(1, Rsb) * besselk(1, Rsa) - besseli(1, Rsa) * besselk(1, Rsb))
    return Zinternal

def calculate_sheath_impedance(sheath_mur, sheath_sig, sheath_inner_radius, sheath_r, outer_radius, Frq,constants):
    """
    【函数功能】套管阻抗计算
    【入参】
    sheath_mur (float): 套管的磁导率
    sheath_sig (float): 套管的电导率
    sheath_inner_radius (float): 套管的内径
    sheath_r (float): 套管的外径
    Frq(numpy.ndarray,1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zs(numpy.ndarray:1*1*Nf): Nf个频率下的套管阻抗矩阵
    """
    mu0, ep0 = constants.mu0, constants.ep0
    Besl_Max = 200
    frq = np.array([Frq]).reshape(-1)
    Mu_s = mu0 * sheath_mur
    Nf = frq.size
    omega = 2 * np.pi * frq
    gamma_s = np.sqrt(1j * Mu_s * omega * (sheath_sig + 1j * omega * ep0))
    Rsa = sheath_inner_radius * gamma_s
    Rsb = sheath_r * gamma_s
    ks = 1j * omega * Mu_s / (2 * np.pi * Rsb)
    # Zs_diag = np.copy(ks)
    #
    # dR = Rsb - Rsa
    # low = np.where(np.real(dR) <= Besl_Max)
    # Zs_diag[low] = ks[low] * np.cosh(dR[low]) / np.sinh(dR[low])
    #
    # low = np.where(np.real(Rsb) <= Besl_Max)
    # tmp1 = besseli(0, Rsb[low]) * besselk(1, Rsa[low]) + besseli(1, Rsa[low]) * besselk(0, Rsb[low])
    # tmp2 = besseli(1, Rsb[low]) * besselk(1, Rsa[low]) - besseli(1, Rsa[low]) * besselk(1, Rsb[low])
    # Zs_diag[low] = ks[low] * tmp1 / tmp2
    tmp1 = besseli(0, Rsb) * besselk(1, Rsa) + besseli(1, Rsa) * besselk(0, Rsb)
    tmp2 = besseli(1, Rsb) * besselk(1, Rsa) - besseli(1, Rsa) * besselk(1, Rsb)
    Zs_diag = ks * tmp1 / tmp2

    Ns = np.array([sheath_sig]).reshape(-1).shape[0]
    Zs = np.zeros((Ns, 1, Nf), dtype='complex')
    for ik in range(Nf):
        Zs[:, 0, ik] = Zs_diag[ik] + ks[ik] * Rsb * np.log(outer_radius/sheath_r)
    # process the data, because the Z matrix is 3-dimensional matrix, but when fre is a float, we need Z is a 2-dimensional array
    # if isinstance(Frq, float):
    #     Zs = np.squeeze(Zs)
    return Zs


def calculate_multual_impedance(core_wires_r, sheath_mur, sheath_sig, sheath_inner_radius, sheath_r, sheath_epr, Frq, constants):
    """
    【函数功能】互阻抗计算
    【入参】
    core_wires_r (numpy.ndarray, n*1): n条芯线的半径
    sheath_mur (float): 套管的磁导率
    sheath_sig (float): 套管的电导率
    sheath_inner_radius (float): 套管的内径
    sheath_r (float): 套管的外径
    Frq(numpy.ndarray,1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zcs(numpy.ndarray:n*1*Nf): Nf个频率下的芯线和表皮之间的互阻抗矩阵, n为芯线数量
    Zsc(numpy.ndarray:1*n*Nf): Nf个频率下的表皮和芯线之间的互阻抗矩阵, n为芯线数量
    """
    mu0, ep0 = constants.mu0, constants.ep0
    Besl_Max = 200
    frq = np.array([Frq]).reshape(-1)
    # Npha 表示芯线数量
    Npha = core_wires_r.shape[0]
    Mu_s = mu0 * sheath_mur
    Epr_s = ep0*sheath_epr
    Nf = frq.size
    omega = 2 * np.pi * frq
    gamma_s = np.sqrt(1j * Mu_s * omega * (sheath_sig + 1j * omega * Epr_s))
    Rsa = sheath_inner_radius * gamma_s
    Rsb = sheath_r * gamma_s
    ks = 1j * omega * Mu_s / (2 * np.pi * Rsa * Rsb)
    Z0 = np.zeros(Nf, dtype='complex')

    # dR = Rsb - Rsa
    # low = np.where(np.real(dR) <= Besl_Max)
    # Itmp1 = Bessel_IK(Rsa[low], 1, Rsb[low], 1) - Bessel_IK(Rsb[low], 1, Rsa[low], 1)
    # Z0[low] = ks[low] / Itmp1
    #
    # low = np.where(np.real(Rsb) <= Besl_Max)
    # Itmp2 = besseli(1, Rsb[low]) * besselk(1, Rsa[low]) - besseli(1, Rsa[low]) * besselk(1, Rsb[low])
    # Z0[low] = ks[low] / Itmp2

    Itmp2 = besseli(1, Rsb) * besselk(1, Rsa) - besseli(1, Rsa) * besselk(1, Rsb)
    Z0 = - ks / Itmp2

    Zcs = np.zeros((Npha, 1, Nf), dtype='complex')
    Zsc = np.zeros((1, Npha, Nf), dtype='complex')
    for ik in range(Nf):
        Zcs[:, 0, ik] = Z0[ik]
        Zsc[0, :, ik] = Z0[ik]
    # process the data, because the Z matrix is 3-dimensional matrix, but when fre is a float, we need Z is a 2-dimensional array
    # if isinstance(Frq, float):
    #     Zsc = np.squeeze(Zsc) # Zsc should be 1*n
    #     Zcs = np.squeeze(Zcs).reshape(-1, 1) # Zcs should be n*1

    return Zcs, Zsc


def calculate_ground_impedance(ground_mur, ground_epr, ground_sig, end_node_z, sheath_outer_radius, Dist, Frq,
                               constants):
    """
    【函数功能】地阻抗计算
    【入参】
    ground_mur(float):大地相对磁导率
    ground_epr(float):大地相对介电常数
    ground_sig(float):大地电导率
    end_node_z (numpy.ndarray,n*1): n条芯线的第二个节点的z值
    sheath_outer_radius (float): 整体外径
    Dist:未知
    Frq(numpy.ndarray,1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zg(numpy.ndarray:1*1*Nf): Nf个频率下的地阻抗矩阵
    """
    mu0, ep0 = constants.mu0, constants.ep0
    Mur_g = ground_mur * mu0
    Epr_g = ground_epr * ep0
    frq = np.array([Frq]).reshape(-1)
    r0 = np.array([sheath_outer_radius]).reshape(-1)
    Ncon = r0.size
    Nf = frq.size
    Zg = np.zeros((Ncon, Ncon, Nf), dtype='complex')

    omega = 2 * np.pi * frq
    gamma = np.sqrt(1j * Mur_g * omega * (ground_sig + 1j * omega * Epr_g))
    km = 1j * omega * Mur_g / 4 / np.pi

    if end_node_z[0] > 0 and end_node_z[0] < 1e6:
        for i1 in range(Ncon):
            for i2 in range(Ncon - i1):
                d = abs(Dist[i1] - Dist[i2 + i1])
                h1 = end_node_z[0][i1]
                h2 = end_node_z[0][i2 + i1]
                Zg[i1, i2 + i1, :] = km * np.log(((1 + gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2) / (
                        (gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2))
                Zg[i2 + i1, i1, :] = np.copy(Zg[i1, i2 + i1, :])

        for i in range(Ncon):
            h = end_node_z[0][i]
            Zg[i, i, :] = km * np.log(((1 + gamma * h) ** 2) / ((gamma * h) ** 2))

    elif end_node_z[0] < 0:
        for i1 in range(Ncon):
            R0 = r0 * gamma
            Zg[i1, i1, :] = 2 * km * np.log((1 + R0) / R0)

    return Zg


def calculate_OHL_wire_impedance(radius, mur, sig, epr, constants, frq=frq_default):
    """
    【函数功能】架空线阻抗参数计算
    【入参】
    radius (numpy.ndarray,n*1): n条线线的半径
    sig (numpy.ndarray,n*1): n条线线的电导率
    mur (numpy.ndarray,n*1): n条线线的磁导率
    epr (numpy.ndarray,n*1): n条线线的相对介电常数
    constants(Constant类)：常数类
    frq(numpy.ndarray，1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zc(numpy.ndarray:n*n*Nf)：n条线在Nf个频率下的阻抗矩阵
    """
    ep0, mu0 = constants.ep0, constants.mu0
    Emax = 350
    Ncon = np.array([radius]).reshape(-1).shape[0]
    Nf = np.array([frq]).reshape(-1).shape[0]
    Zc = np.zeros((Ncon, Ncon, Nf), dtype='complex')
    omega = 2 * np.pi * frq
    for i in range(Nf):
        gamma = np.sqrt(1j * mu0 * mur * omega[i] * (sig + 1j * omega[i] * ep0 * epr))
        Ri = radius * gamma
        I0i = besseli(0, Ri)
        I1i = besseli(1, Ri)
        out = gamma / (2 * np.pi * radius * sig)
        low = np.where(abs(Ri) < Emax)
        out[low] = 1j * mu0 * mur[low] * omega[i] * I0i[low] / (2 * np.pi * Ri[low] * I1i[low])
        Zc[:, :, i] = np.diag(out.reshape((-1)))
    return Zc


def calculate_OHL_ground_impedance(sig_g, mur_g, epr_g, radius, offset, height, constants, frq=frq_default):
    """
    【函数功能】架空线大地阻抗参数计算
    【入参】
    offset (numpy.ndarray,n*1): n条线的偏置
    radius (numpy.ndarray,n*1): n条线的半径
    sig_g (float): 大地的电导率
    mur_g (float): 大地的磁导率
    epr_g (float): 大地的相对介电常数
    constants(Constant类)：常数类
    frq(numpy.ndarray，1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Zg(numpy.ndarray:n*n*Nf)：n条线对应的大地阻抗矩阵
    """
    ep0, mu0 = constants.ep0, constants.mu0
    Sig_g = sig_g
    Mur_g = mur_g * mu0
    Eps_g = epr_g * ep0
    Ncon = np.array([radius]).reshape(-1).shape[0]
    Nf = np.array([frq]).reshape(-1).shape[0]
    Zg = np.zeros((Ncon, Ncon, Nf), dtype='complex')
    omega = 2 * np.pi * frq
    gamma = np.sqrt(1j * Mur_g * omega * (Sig_g + 1j * omega * Eps_g))
    km = 1j * omega * Mur_g / 4 / np.pi
    for i in range(Ncon):
        for j in range(i, Ncon):
            d = abs(offset[i] - offset[j])
            h1 = height[i]
            h2 = height[j]
            Zg[i, j, :] = km * np.log(((1 + gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2) / (
                    (gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2))
            Zg[j, i, :] = np.copy(Zg[i, j, :])
    for i in range(Ncon):
        h = height[i]
        Zg[i, i, :] = km * np.log(((1 + gamma * h) ** 2) / ((gamma * h) ** 2))
    return Zg


def calculate_OHL_impedance(radius, mur, sig, epr, offset, height, sig_g, mur_g, epr_g, Lm, constants, frq=frq_default):
    """
    【函数功能】阻抗矩阵参数计算
    【入参】
    height(numpy.ndarray,n*1):n条线高
    offset (numpy.ndarray,n*1): n条线的偏置
    radius (numpy.ndarray,n*1): n条线的半径
    sig (numpy.ndarray,n*1): n条线的电导率
    mur (numpy.ndarray,n*1): n条线的磁导率
    epr (numpy.ndarray,n*1): 线n条的相对介电常数
    sig_g (float): 大地的电导率
    mur_g (float): 大地的磁导率
    epr_g (float): 大地的相对介电常数
    Lm(numpy.ndarray:n*n)：n条线的互感矩阵
    constants(Constant类)：常数类
    frq(numpy.ndarray，1*Nf):Nf个频率组成的频率矩阵

    【出参】
    Z(numpy.ndarray:n*n)：n条线的阻抗矩阵
    """
    Zc = calculate_OHL_wire_impedance(radius, mur, sig, epr, constants, frq)
    Zg = calculate_OHL_ground_impedance(sig_g, mur_g, epr_g, radius, offset, height, constants, frq)
    Nf = np.array([frq]).reshape(-1).shape[0]
    Z = Zc + Zg
    for i in range(Nf):
        Z[:, :, i] += 1j * 2 * np.pi * frq[i] * Lm
    return Z