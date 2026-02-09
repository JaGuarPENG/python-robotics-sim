"""
示教器主窗口 (重构后 - 组件化架构)
"""

import threading
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QMessageBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer

from fast_ik.ik_solver import FastIKSolver
from .signals import WorkerSignals
from .robot_3d_widget import Robot3DWidget
from .robot_controller import RobotController

# 导入 UI 组件
from .ui.connection_panel import ConnectionPanel
from .ui.robot_status_panel import RobotStatusPanel
from .ui.joint_control_panel import JointControlPanel
from .ui.teleop_panel import TeleopPanel
from .ui.follower_panel import FollowerPanel

# 导入逻辑服务
from .logic.trajectory_service import TrajectoryService

class TeachPendantWindow(QMainWindow):
    """示教器主窗口"""

    CONTINUOUS_TEST_POINTS = [
        (488, 164, 957, 225, -20, 0), # (x, y, z, rx, ry, rz) mm, deg
    ]

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.controller = RobotController(self.signals)
        self.fast_ik = FastIKSolver()
        
        # 初始化逻辑服务
        self.traj_service = TrajectoryService(self.controller, self.signals, self.fast_ik)

        self.init_ui()
        self.connect_signals()

        # 启动 3D 视图更新定时器 (20Hz)
        self.view_timer = QTimer()
        self.view_timer.timeout.connect(self.update_3d_view)
        self.view_timer.start(50)

    def init_ui(self):
        """初始化 UI"""
        self.setWindowTitle("机器人示教器 - Teach Pendant (Refactored)")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ==========================================
        # 左侧：3D 可视化与状态
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 1. 3D 视图
        view_group = QGroupBox("3D 机械臂视图")
        view_layout = QVBoxLayout(view_group)
        self.robot_view = Robot3DWidget()
        self.robot_view.set_test_waypoints(self.CONTINUOUS_TEST_POINTS)
        view_layout.addWidget(self.robot_view)

        view_btn_layout = QHBoxLayout()
        reset_view_btn = QPushButton("重置视角")
        reset_view_btn.clicked.connect(self.robot_view.reset_view)
        view_btn_layout.addWidget(reset_view_btn)
        
        self.move_target_btn = QPushButton("移动到红点 (IK)")
        self.move_target_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.move_target_btn.clicked.connect(self.on_move_to_target)
        view_btn_layout.addWidget(self.move_target_btn)
        
        self.run_traj_btn = QPushButton("画圆轨迹 (Trajectory)")
        self.run_traj_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")
        self.run_traj_btn.clicked.connect(self.on_run_trajectory)
        view_btn_layout.addWidget(self.run_traj_btn)
        
        view_btn_layout.addStretch()
        view_layout.addLayout(view_btn_layout)
        left_layout.addWidget(view_group)

        # 2. 状态显示 (TCP & Robot Status)
        self.status_panel = RobotStatusPanel(self.controller, self.signals)
        self.status_panel.model_toggle_requested.connect(self.on_toggle_model)
        left_layout.addWidget(self.status_panel)

        main_layout.addWidget(left_panel, stretch=3)

        # ==========================================
        # 右侧：控制面板
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 1. 连接控制
        self.conn_panel = ConnectionPanel(self.controller, self.signals)
        right_layout.addWidget(self.conn_panel)

        # 2. 跟随模式
        self.follower_panel = FollowerPanel(self.controller, self.signals)
        # 一键启动按钮需要主窗口协助（获取 IP）
        self.follower_panel.one_click_follower_btn.disconnect()
        self.follower_panel.one_click_follower_btn.clicked.connect(self.on_one_click_follower)
        right_layout.addWidget(self.follower_panel)

        # 3. 关节控制
        self.joint_control_panel = JointControlPanel(self.controller, self.signals)
        right_layout.addWidget(self.joint_control_panel)

        # 4. 遥操作测试
        self.teleop_panel = TeleopPanel(self.controller, self.signals)
        right_layout.addWidget(self.teleop_panel)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=2)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("请先连接机器人")

        # 全局样式
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 2px solid #3d3d3d; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { min-height: 25px; border-radius: 4px; }
            QPushButton:disabled { background-color: #555; color: #999; }
        """)

    def connect_signals(self):
        """连接全局信号"""
        self.signals.status_updated.connect(self.update_status) # 修正：连接到自己的 update_status
        self.signals.error_occurred.connect(lambda msg: QMessageBox.critical(self, "错误", msg))
        self.signals.connection_changed.connect(self.on_connection_changed)
        self.signals.command_finished.connect(self.on_command_finished)

    def update_status(self, message):
        """处理状态更新，激活 UI 组件"""
        self.statusBar.showMessage(message)

        if "登录成功" in message:
            self.status_panel.set_refresh_enabled(True)

        if "已退出登录" in message:
            self.status_panel.set_refresh_enabled(False)
            self.joint_control_panel.enable_controls(False)
            self.follower_panel.set_controls_enabled(False)
            self.teleop_panel.set_udp_enabled(False)

        if "使能成功" in message:
            # 核心激活点
            self.joint_control_panel.enable_controls(True)
            self.follower_panel.set_controls_enabled(True)
            self.teleop_panel.set_udp_enabled(True)

        if "跟随模式" in message or "follower_cart" in message:
            # 确保在控制器状态更新后，强制刷新遥操作面板按钮
            self.teleop_panel._update_test_btns()

    def update_3d_view(self):
        """同步更新 3D 视图与 TCP 显示"""
        # 直接通过控制器属性获取数据 (内部已处理线程安全)
        joints = self.controller.current_joints
        actual_tcp = self.controller.actual_tcp

        self.robot_view.update_robot(joints)

        if actual_tcp:
            self.status_panel.update_tcp_display(actual_tcp=actual_tcp)
        else:
            try:
                T = self.robot_view.robot.fkine(np.deg2rad(joints))
                self.status_panel.update_tcp_display(model_pos=T.t*1000, model_rpy=T.rpy(unit='deg', order='zyx'))
            except: pass

    def on_connection_changed(self, connected, port_type):
        """处理连接状态变化"""
        if port_type == "websocket" and connected:
            self.statusBar.showMessage("WebSocket 已连接")

    def on_command_finished(self, success, message):
        """通用命令结果处理"""
        # 修正：通过 update_status 统一处理，以激活相关 UI
        self.update_status(message)

    def on_one_click_follower(self):
        """中转：调用 FollowerPanel 并传入当前 IP"""
        ip = self.conn_panel.get_ip()
        self.follower_panel.on_one_click_follower(ip)

    def on_toggle_model(self):
        """切换 3D 机器人模型显示"""
        mode_name = self.robot_view.toggle_render_mode()
        self.statusBar.showMessage(f"当前显示模式: {mode_name}")

    def on_move_to_target(self):
        """移动到 3D 目标点 (逻辑委托给 traj_service)"""
        target = self.CONTINUOUS_TEST_POINTS[0]
        self.statusBar.showMessage("正在解算 IK...")
        
        target_joints, ik_time = self.traj_service.move_to_target(target, self.controller.current_joints)
        
        if target_joints:
            msg = f"IK 解算成功 ({ik_time:.2f} ms)，是否立即移动?\n{['%.2f'%j for j in target_joints]}"
            if QMessageBox.question(self, "确认移动", msg) == QMessageBox.Yes:
                self.controller.move_joint(target_joints, vels=[20, 20, 20])
        else:
            QMessageBox.warning(self, "失败", f"无法找到逆解 ({ik_time:.2f} ms)")

    def on_run_trajectory(self):
        """执行圆形轨迹 (逻辑委托给 traj_service)"""
        center = self.CONTINUOUS_TEST_POINTS[0]
        radius, num_points = 50.0, 12
        
        if QMessageBox.question(self, "确认执行", f"即将执行圆形轨迹 (半径 {radius}mm, {num_points}点)，是否开始?") == QMessageBox.Yes:
            points = self.traj_service.run_circular_trajectory(center, radius, num_points)
            self.robot_view.set_trajectory(points)

    def closeEvent(self, event):
        """清理资源"""
        self.view_timer.stop()
        self.controller.stop()
        if hasattr(self, 'fast_ik'): self.fast_ik.close()
        event.accept()