"""
3D 机械臂可视化控件 (重构后 - 基于组件化渲染)
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from pyvistaqt import QtInteractor

from .render.robot_model import RobotModel
from .render.robot_renderer import RobotRenderer

class Robot3DWidget(QWidget):
    """3D 机械臂可视化控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. 核心渲染引擎
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # 2. 逻辑与渲染组件
        self.model = RobotModel()
        self.renderer = RobotRenderer(self.plotter, self.model)

        self.setup_scene()
        self.renderer.create_robot_actors()
        self.update_robot(np.zeros(6))

    def setup_scene(self):
        """设置 3D 场景基础环境"""
        self.plotter.set_background('#1a1a2e', top='#16213e')
        self.renderer.setup_base_scene()
        
        # 相机初始位姿
        self.plotter.camera_position = [(3.68, -2.21, 3.34), (0.00, 0.00, 0.30), (-0.43, 0.41, 0.81)]
        self.plotter.enable_anti_aliasing()

    def update_robot(self, joints_deg):
        """外部接口：更新机器人姿态"""
        self.renderer.update(np.deg2rad(joints_deg))

    def set_trajectory(self, points_mm):
        """外部接口：设置轨迹线"""
        self.renderer.set_trajectory(points_mm)

    def clear_trajectory(self):
        self.renderer.clear_trajectory()

    def toggle_render_mode(self):
        """切换简化/精细模型"""
        new_mode = 1 - self.renderer.mode
        self.renderer.set_mode(new_mode)
        # 立即根据当前角度重绘一次
        self.update_robot(self.model.dh_robot.q * 180 / np.pi)
        return "精细模型" if new_mode == 1 else "简化模型"

    def reset_view(self):
        self.plotter.camera_position = [(3.68, -2.21, 3.34), (0.00, 0.00, 0.30), (-0.43, 0.41, 0.81)]
        self.plotter.reset_camera()

    def set_test_waypoints(self, waypoints):
        """设置参考测试点位 (简化版，仅作占位)"""
        # 如果需要更复杂的点位标记，可在 renderer 中添加相应方法
        pass

    @property
    def robot(self):
        """保留对底层 DH 模型的访问，兼顾向后兼容性"""
        return self.model.robot

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)