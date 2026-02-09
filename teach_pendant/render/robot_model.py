"""
机器人几何模型与运动学计算 (渲染专用)
"""

import numpy as np
from tools.robot_dh import create_ka_ur

class RobotModel:
    """机器人几何属性与 FK 封装"""

    def __init__(self):
        self.robot = create_ka_ur()
        self.link_colors = [
            '#E74C3C', '#F39C12', '#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C', '#E74C3C'
        ]

    def get_joint_positions(self, joints_rad):
        """计算所有关节在世界坐标系中的 3D 位置 (米)"""
        positions = [[0, 0, 0]]  # Base
        for i in range(len(joints_rad)):
            T = self.robot.fkine(joints_rad, end=self.robot.links[i])
            positions.append(T.t.tolist())
        
        # End-effector
        T_end = self.robot.fkine(joints_rad)
        positions.append(T_end.t.tolist())
        return np.array(positions)

    def get_ee_pose(self, joints_rad):
        """获取末端执行器的 SE3 位姿"""
        T = self.robot.fkine(joints_rad)
        return T.t, T.R
