"""
轨迹执行服务逻辑 (IK 解算与运动控制循环)
"""

import threading
import time
import numpy as np
from PyQt5.QtWidgets import QMessageBox

class TrajectoryService:
    """轨迹执行服务"""

    def __init__(self, controller, signals, fast_ik):
        self.controller = controller
        self.signals = signals
        self.fast_ik = fast_ik

    def run_circular_trajectory(self, center, radius, num_points):
        """执行圆形轨迹"""
        if not self.controller.state.is_enabled:
            self.signals.error_occurred.emit("机器人未使能，无法执行轨迹")
            return []

        cx, cy, cz, rx, ry, rz = center
        
        trajectory_points = []
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        angles = np.append(angles, 0) # 闭合
        
        for theta in angles:
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = cz
            trajectory_points.append((x, y, z))
            
        def run_task():
            vels = [15, 15, 15]
            last_q_rad = np.deg2rad(self.controller.current_joints)
            
            total = len(trajectory_points)
            for i, (tx, ty, tz) in enumerate(trajectory_points):
                self.signals.status_updated.emit(f"正在前往轨迹点 {i+1}/{total}...")
                
                target_pos = [tx/1000.0, ty/1000.0, tz/1000.0]
                target_rpy = [rx, ry, rz]
                
                target_q_rad, ik_time = self.fast_ik.solve_ik(
                    pos=target_pos,
                    rpy_deg=target_rpy,
                    current_joints=last_q_rad
                )
                
                if target_q_rad is not None:
                    target_joints_deg = np.rad2deg(target_q_rad).tolist()
                    last_q_rad = target_q_rad
                    self.controller.move_joint(target_joints_deg, vels)
                else:
                    self.signals.error_occurred.emit(f"轨迹点 {i+1} IK解算失败!")
                    break
            
            self.signals.status_updated.emit("轨迹执行完成")
            
        threading.Thread(target=run_task, daemon=True).start()
        return trajectory_points

    def move_to_target(self, target, current_joints):
        """移动到单个目标点 (IK)"""
        x, y, z, rx, ry, rz = target
        pos = [x/1000.0, y/1000.0, z/1000.0]
        rpy = [rx, ry, rz]
        current_q_rad = np.deg2rad(current_joints)
        
        target_q_rad, ik_time = self.fast_ik.solve_ik(
            pos=pos, 
            rpy_deg=rpy, 
            current_joints=current_q_rad
        )
        
        if target_q_rad is None:
            return None, ik_time
            
        target_joints_deg = np.rad2deg(target_q_rad).tolist()
        return target_joints_deg, ik_time
