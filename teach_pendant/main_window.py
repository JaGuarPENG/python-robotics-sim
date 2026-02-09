"""
示教器主窗口 (全功能恢复版 - 全屏、大字号、性能监控)
"""

import threading
import time
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QMessageBox, QStatusBar, QTabWidget, QLabel
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
from .ui.vision_panel import VisionPanel

# 导入逻辑服务
from .logic.trajectory_service import TrajectoryService
import config

class TeachPendantWindow(QMainWindow):
    """示教器主窗口 - 工业级全屏界面"""

    CONTINUOUS_TEST_POINTS = [
        (488, 164, 957, 180, 0, 0), # 示例点 (mm, deg)
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

        # 核心缓存
        self.current_vision_trajectory = None
        self.actual_trajectory_log = []
        self.is_recording_actual = False
        self.current_target_point = None
        self.playback_frequency = 30.0 # 默认频率

        # 启动 3D 视图更新定时器 (30Hz)
        self.view_timer = QTimer()
        self.view_timer.timeout.connect(self.update_3d_view)
        self.view_timer.start(33)

    def init_ui(self):
        """初始化 UI 布局"""
        self.setWindowTitle("机器人示教器 - Teach Pendant (Industrial FS)")
        self.setMinimumSize(1400, 900)
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)

        # ==========================================
        # 左侧：3D 预览与实时状态
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        view_group = QGroupBox("3D 机械臂监控系统")
        view_layout = QVBoxLayout(view_group)
        self.robot_view = Robot3DWidget()
        view_layout.addWidget(self.robot_view)

        view_btn_layout = QHBoxLayout()
        reset_view_btn = QPushButton("重置视角")
        reset_view_btn.clicked.connect(self.robot_view.reset_view)
        view_btn_layout.addWidget(reset_view_btn)
        
        self.move_target_btn = QPushButton("移动至目标点 (IK)")
        self.move_target_btn.setStyleSheet("background-color: #e67e22; color: white;")
        self.move_target_btn.clicked.connect(self.on_move_to_target)
        view_btn_layout.addWidget(self.move_target_btn)
        
        self.run_traj_btn = QPushButton("画圆轨迹 (Circle)")
        self.run_traj_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.run_traj_btn.clicked.connect(self.on_run_trajectory)
        view_btn_layout.addWidget(self.run_traj_btn)
        
        view_btn_layout.addStretch()
        view_layout.addLayout(view_btn_layout)
        left_layout.addWidget(view_group)

        # 状态面板 (包含线性偏差)
        self.status_panel = RobotStatusPanel(self.controller, self.signals)
        left_layout.addWidget(self.status_panel)

        main_layout.addWidget(left_panel, stretch=4)

        # ==========================================
        # 右侧：分类功能选项卡
        # ==========================================
        self.tabs = QTabWidget()
        
        # Tab 1: 基础控制
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        self.conn_panel = ConnectionPanel(self.controller, self.signals)
        self.joint_control_panel = JointControlPanel(self.controller, self.signals)
        basic_layout.addWidget(self.conn_panel)
        basic_layout.addWidget(self.joint_control_panel)
        basic_layout.addStretch()
        self.tabs.addTab(basic_tab, "基础控制")

        # Tab 2: 高级运动
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        self.follower_panel = FollowerPanel(self.controller, self.signals)
        self.follower_panel.one_click_follower_btn.disconnect()
        self.follower_panel.one_click_follower_btn.clicked.connect(self.on_one_click_follower)
        self.teleop_panel = TeleopPanel(self.controller, self.signals)
        advanced_layout.addWidget(self.follower_panel)
        advanced_layout.addWidget(self.teleop_panel)
        advanced_layout.addStretch()
        self.tabs.addTab(advanced_tab, "高级运动")

        # Tab 3: 视觉引导
        vision_tab = QWidget()
        vision_layout = QVBoxLayout(vision_tab)
        self.vision_panel = VisionPanel()
        self.vision_panel.trajectory_generated.connect(self.on_vision_trajectory_ready)
        self.vision_panel.execution_requested.connect(self.on_execute_vision_trajectory)
        self.vision_panel.actual_export_requested.connect(self.on_save_actual_trajectory)
        vision_layout.addWidget(self.vision_panel)
        vision_layout.addStretch()
        self.tabs.addTab(vision_tab, "视觉引导")

        main_layout.addWidget(self.tabs, stretch=2)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统就绪，请连接机器人")

        # 全局大字号样式重构
        self.setStyleSheet("""
            QWidget { font-size: 24px; background-color: #1e1e1e; color: #ecf0f1; }
            QGroupBox { font-size: 26px; font-weight: bold; border: 3px solid #3d3d3d; border-radius: 12px; margin-top: 25px; padding-top: 25px; }
            QGroupBox::title { subcontrol-origin: margin; left: 20px; padding: 0 10px; }
            QPushButton { min-height: 65px; border-radius: 10px; font-size: 24px; padding: 5px 20px; }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QLabel { color: #ecf0f1; font-size: 24px; }
            QTabBar::tab { min-width: 200px; min-height: 70px; font-size: 24px; font-weight: bold; }
            QTabBar::tab:selected { background: #3498db; color: white; }
            QTabWidget::pane { border: 3px solid #3d3d3d; border-radius: 8px; }
        """)

    def connect_signals(self):
        """连接所有全局信号，补全状态联动"""
        self.signals.status_updated.connect(self.update_status)
        self.signals.error_occurred.connect(lambda msg: QMessageBox.critical(self, "系统错误", msg))
        self.signals.connection_changed.connect(self.on_connection_changed)
        self.signals.command_finished.connect(self.on_command_finished)

    def update_status(self, message):
        """状态更新机：处理 UI 激活与频率更改"""
        # 1. 处理频率更改指令
        if message.startswith("FREQ_CHANGE:"):
            try:
                hz = int(message.split(":")[1])
                self.playback_frequency = float(hz)
                self.statusBar.showMessage(f"已切换播放频率至: {hz} Hz")
                return
            except: pass

        self.statusBar.showMessage(message)

        # 2. 核心状态 UI 联动
        if "登录成功" in message:
            self.status_panel.set_refresh_enabled(True)

        if "已退出登录" in message:
            self.status_panel.set_refresh_enabled(False)
            self.joint_control_panel.enable_controls(False)
            self.follower_panel.set_controls_enabled(False)
            self.teleop_panel.set_udp_enabled(False)

        if "使能成功" in message:
            self.joint_control_panel.enable_controls(True)
            self.follower_panel.set_controls_enabled(True)
            self.teleop_panel.set_udp_enabled(True)

        if "跟随模式" in message or "follower_cart" in message:
            self.teleop_panel._update_test_btns()

    def update_3d_view(self):
        """同步更新 3D 与 误差计算"""
        joints = self.controller.current_joints
        actual_tcp = self.controller.actual_tcp
        self.robot_view.update_robot(joints)

        if actual_tcp:
            self.status_panel.update_tcp_display(actual_tcp=actual_tcp)
            # 记录轨迹
            if self.is_recording_actual:
                self.actual_trajectory_log.append(list(actual_tcp))
                self.robot_view.renderer.add_actual_point(np.array(actual_tcp[:3])/1000.0)
            
            # 计算线性偏差 (扣除 50mm Z 轴偏差)
            if self.current_target_point:
                p = self.current_target_point
                actual_fixed = np.array([actual_tcp[0], actual_tcp[1], actual_tcp[2] - 50.0])
                lin_err = np.linalg.norm(np.array(p[:3]) - actual_fixed)
                self.signals.tracking_error_updated.emit(lin_err, 0.0)
        else:
            # 回退到 DH 计算预览
            try:
                T = self.robot_view.robot.fkine(np.deg2rad(joints))
                self.status_panel.update_tcp_display(model_pos=T.t*1000, model_rpy=T.rpy(unit='deg', order='zyx'))
            except: pass

    def on_vision_trajectory_ready(self, points):
        self.current_vision_trajectory = points
        display_points = [(p[0], p[1], p[2]) for p in points]
        self.robot_view.set_trajectory(display_points)
        self.statusBar.showMessage(f"轨迹加载成功，共 {len(points)} 个点位")

    def on_execute_vision_trajectory(self):
        if not self.current_vision_trajectory:
            QMessageBox.warning(self, "提醒", "请先加载轨迹数据")
            return
        
        self.actual_trajectory_log = []
        self.robot_view.renderer.clear_actual_path()
        self.is_recording_actual = True
            
        if QMessageBox.question(self, "确认执行", f"即将以 {self.playback_frequency}Hz 速率执行轨迹，是否开始?") == QMessageBox.Yes:
            threading.Thread(target=self._run_points_sequence, args=(self.current_vision_trajectory,), daemon=True).start()
        else:
            self.is_recording_actual = False

    def _run_points_sequence(self, points):
        """核心轨迹执行算法 (包含实时解算、异步下发与性能计时)"""
        print(f"\n轨迹执行性能报告 (Hz: {self.playback_frequency})")
        print(f"{'点位':<8} | {'IK耗时(ms)':<12} | {'指令耗时(ms)':<12} | {'总周期(ms)':<12}")
        print("-" * 60)
        
        target_interval = 1.0 / self.playback_frequency
        try:
            last_q = self.controller.current_joints
            for i, p in enumerate(points):
                t_loop_start = time.perf_counter()
                if not self.controller.state.is_enabled: break
                
                self.current_target_point = p
                
                # 1. 实时 IK 解算
                t_ik = time.perf_counter()
                pos_m = [p[0]/1000.0, p[1]/1000.0, p[2]/1000.0]
                q_rad, _ = self.fast_ik.solve_ik(pos=pos_m, rpy_deg=p[3:], current_joints=np.deg2rad(last_q))
                dt_ik = (time.perf_counter() - t_ik) * 1000
                
                # 2. 高速异步指令发送
                t_cmd = time.perf_counter()
                if q_rad:
                    self.controller.move_joint(np.rad2deg(q_rad).tolist(), vels=[100,100,100], wait_for_finish=False)
                    last_q = np.rad2deg(q_rad)
                dt_cmd = (time.perf_counter() - t_cmd) * 1000
                
                dt_total = (time.perf_counter() - t_loop_start) * 1000
                if i % 10 == 0:
                    print(f"{i:<8} | {dt_ik:<12.2f} | {dt_cmd:<12.2f} | {dt_total:<12.2f}")
                
                # 3. 精准频率控制
                time.sleep(max(0, target_interval - (time.perf_counter() - t_loop_start)))
            
            self.is_recording_actual = False
            self.signals.status_updated.emit("轨迹执行完成")
        except Exception as e:
            self.signals.error_occurred.emit(f"执行异常: {e}")

    def on_save_actual_trajectory(self):
        if not self.actual_trajectory_log: return
        try:
            import pandas as pd
            import os
            os.makedirs('csv', exist_ok=True)
            pd.DataFrame(self.actual_trajectory_log, columns=['x','y','z','rx','ry','rz']).to_csv('csv/actual_trajectory.csv', index=False)
            QMessageBox.information(self, "导出成功", "实际运行轨迹已保存至 csv/actual_trajectory.csv")
        except Exception as e: QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def on_one_click_follower(self):
        ip = self.conn_panel.get_ip()
        self.follower_panel.on_one_click_follower(ip)

    def on_move_to_target(self):
        target = self.CONTINUOUS_TEST_POINTS[0]
        self.statusBar.showMessage("正在解算 IK...")
        target_joints, ik_time = self.traj_service.move_to_target(target, self.controller.current_joints)
        if target_joints:
            if QMessageBox.question(self, "确认移动", f"IK 解算成功({ik_time:.1f}ms)，是否移动？") == QMessageBox.Yes:
                self.controller.move_joint(target_joints, vels=[30,30,30])
        else: QMessageBox.warning(self, "错误", "无法找到逆解")

    def on_run_trajectory(self):
        center = self.CONTINUOUS_TEST_POINTS[0]
        points = self.traj_service.run_circular_trajectory(center, 50.0, 36)
        self.robot_view.set_trajectory(points)

    def on_command_finished(self, success, message): self.update_status(message)
    def on_connection_changed(self, connected, port_type): 
        self.statusBar.showMessage(f"{port_type} 已{'连接' if connected else '断开'}")

    def closeEvent(self, event):
        self.view_timer.stop()
        self.controller.stop()
        self.fast_ik.close()
        event.accept()
