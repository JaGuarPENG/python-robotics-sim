"""
机器人几何模型与运动学计算 (支持多模型加载)
"""

import roboticstoolbox as rtb
import os
import numpy as np
from tools.robot_dh import create_ka_ur

class RobotModel:
    """机器人几何属性与运动学封装"""

    def __init__(self):
        # 1. 加载简化 DH 模型
        self.dh_robot = create_ka_ur()
        
        # 2. 加载精细 URDF 模型
        self.urdf_robot = None
        self.load_urdf()
        
        self.link_colors = [
            '#E74C3C', '#F39C12', '#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C', '#E74C3C'
        ]
        
        # STL 映射关系
        self.stl_map = {
            "base_link": "base_link.STL",
            "link1": "link1.STL",
            "link2": "link2.STL",
            "link3": "link3.STL",
            "link4": "link4.STL",
            "link5": "link5.STL",
            "link6": "link6.STL",
        }

    def load_urdf(self):
        """加载 URDF 文件"""
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            urdf_path = os.path.join(current_dir, "urdf_export_fine", "urdf_export_fine", "urdf", "urdf_export_fine.urdf")
            if os.path.exists(urdf_path):
                self.urdf_robot = rtb.ERobot.URDF(urdf_path)
                print("[RobotModel] URDF 模型加载成功")
        except Exception as e:
            print(f"[RobotModel] URDF 加载失败: {e}")

    def get_joint_positions(self, joints_rad):
        """(用于简化模型) 计算所有关节在世界坐标系中的 3D 位置"""
        positions = [[0, 0, 0]]
        for i in range(len(joints_rad)):
            T = self.dh_robot.fkine(joints_rad, end=self.dh_robot.links[i])
            positions.append(T.t.tolist())
        
        T_end = self.dh_robot.fkine(joints_rad)
        positions.append(T_end.t.tolist())
        return np.array(positions)

    def get_ee_pose(self, joints_rad, use_urdf=False):
        """获取末端执行器的 SE3 位姿"""
        robot = self.urdf_robot if (use_urdf and self.urdf_robot) else self.dh_robot
        T = robot.fkine(joints_rad)
        return T.t, T.R