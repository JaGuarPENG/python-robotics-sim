"""
机器人状态容器 - 线程安全
"""

import threading
import numpy as np
from typing import List, Optional

class RobotState:
    """集中管理机器人所有实时状态数据"""

    def __init__(self):
        self._lock = threading.Lock()
        
        # 基础通信状态
        self.is_connected = False
        self.is_logged_in = False
        self.is_enabled = False
        self.is_follower_mode = False
        
        # 位置数据
        self.current_joints = [0.0] * 6  # 度
        self.actual_tcp = [0.0] * 6     # mm, deg (x, y, z, rx, ry, rz)
        
        # 运行状态字典 (来自 get_robot_status)
        self.status_info = {
            'status': '--',
            'activate': '--',
            'motion': '--',
            'mode': '--',
            'error': '无'
        }

    def update_joints(self, joints: List[float]):
        with self._lock:
            self.current_joints = list(joints)

    def get_joints(self) -> List[float]:
        with self._lock:
            return list(self.current_joints)

    def update_tcp(self, tcp: List[float]):
        with self._lock:
            self.actual_tcp = list(tcp)

    def get_tcp(self) -> List[float]:
        with self._lock:
            return list(self.actual_tcp)

    def update_status_info(self, info: dict):
        with self._lock:
            self.status_info.update(info)

    def get_status_info(self) -> dict:
        with self._lock:
            return dict(self.status_info)

    @property
    def joints_np(self) -> np.ndarray:
        """返回 numpy 格式的关节角 (弧度)"""
        return np.deg2rad(self.get_joints())
