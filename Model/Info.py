class Info:
    def __init__(self, name, ID, Type):
        """
        初始化自描述信息对象
        
        参数:
        name (str): 模型的名称
        ID (int): 模型的ID
        Type (str): 模型的类型
        """
        self.name = name
        self.ID = ID
        self.Type = Type


class TowerInfo(Info):

    def __init__(self, Tower_name, Tower_ID, Tower_Type, Tower_position,Vclass, Theta, Mode_Con, Mode_Gnd, Pole_Height, Pole_Head_Node):
        """
        初始化杆塔自描述信息对象
        
        参数:
        name (str): 杆塔名
        ID (int): 序号
        Type (str): 杆塔类型
        Tower_Vclass (str): 杆塔电压等级
        Center_Node (Node(name, x, y, z)): 中心节点
        Theta (float): 杆塔偏转角度
        Mode_Con (int): VF设置
        Mode_Gnd (int): 镜像VF设置
        Pole_Height (float): 杆塔高度
        Pole_Head_Node (Node(name, x, y, z)): 杆塔头节点
        """
        super().__init__(Tower_name, Tower_ID, Tower_Type)
        self.Vclass = Vclass
        self.Theta = Theta
        self.Mode_Con = Mode_Con
        self.Mode_Gnd = Mode_Gnd
        self.Pole_Height = Pole_Height
        self.Pole_Head_Node = Pole_Head_Node
        self.position = Tower_position

class OHLInfo(Info):
    def __init__(self, OHL_name, OHL_ID, OHL_Type, dL, model1, model2,
                 HeadTower,HeadTower_id,HeadTower_pos
                 , TailTower,TailTower_id,TailTower_pos):
        """
        初始化杆塔自描述信息对象
        
        参数:
        name (str): 架空线名称
        ID (int): 架空线序号
        Type (str): 架构线类型
        dL (float): 元线段长度
        model1 (str): 架空线模型1
        model2 (str): 架空线模型2
        HeadTower (str): 架空线头杆塔
        TailTower (str): 架空线尾杆塔

        """
        super().__init__(OHL_name, OHL_ID, OHL_Type)
        self.dL = dL
        self.type = OHL_Type
        self.model1 = model1
        self.model2 = model2
        self.HeadTower = HeadTower
        self.HeadTower_id = HeadTower_id
        self.HeadTower_pos = HeadTower_pos
        self.TailTower = TailTower
        self.TailTower_id = TailTower_id
        self.TailTower_pos = TailTower_pos


class CableInfo(Info):
    def __init__(self, cable_name, cable_ID, cable_Type, HeadTower, T_head_id,T_head_pos,TailTower, T_tail_id,T_tail_pos,
                 core_num, armor_num, delta_L, mode_con, mode_gnd):
        """
        初始化杆塔自描述信息对象

        """
        super().__init__(cable_name, cable_ID, cable_Type)
        self.name = cable_name
        self.id = cable_ID
        self.type = cable_Type
        self.HeadTower = HeadTower
        self.HeadTower_id = T_head_id
        self.HeadTower_pos = T_head_pos
        self.TailTower = TailTower
        self.TailTower_id = T_tail_id
        self.TailTower_pos = T_tail_pos
        self.core_num = core_num
        self.armor_num = armor_num
        self.delta_L = delta_L
        self.mode_con = mode_con
        self.mode_gnd = mode_gnd
