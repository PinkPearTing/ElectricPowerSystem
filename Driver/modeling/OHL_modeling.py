from Function.Calculators.Impedance import calculate_OHL_impedance
from Function.Calculators.Capacitance import calculate_OHL_capcitance
from Function.Calculators.Inductance import calculate_OHL_mutual_inductance, calculate_OHL_inductance
from Function.Calculators.Resistance import calculate_OHL_resistance


def OHL_building(OHL, frequency):
    """
    【函数功能】线路参数计算
    【入参】
    OHL (OHLWire): 管状线段对象
    GND(Ground):大地对象

    【出参】
    R (numpy.ndarray,n*n): n条线的电阻矩阵
    Z(numpy.ndarray:n*n)：n条线的阻抗矩阵
    L(numpy.ndarray:n*n)：n条线的电感矩阵
    C(numpy.ndarray:n*n)：n条线的电容矩阵
    """
    OHL_r = OHL.get_radius()
    OHL_height = OHL.get_height()
    Lm = calculate_OHL_mutual_inductance(OHL_r, OHL_height, OHL.get_end_node_y())

    L = calculate_OHL_inductance(OHL.get_inductance(), Lm)

    C = calculate_OHL_capcitance(Lm)

    Z = calculate_OHL_impedance(OHL_r, OHL.get_mur(), OHL.get_sig(), OHL.get_epr(), OHL.get_offset(), OHL_height,
                                OHL.ground.sig, OHL.ground.mur, OHL.ground.epr, Lm, frequency)

    R = calculate_OHL_resistance(OHL.get_resistance())
    return R, L, Z, C