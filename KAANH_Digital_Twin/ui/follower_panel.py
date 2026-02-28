"""
跟随模式控制面板组件
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton, QFrame, QHBoxLayout
)
from PyQt5.QtCore import Qt

from ..signals import WorkerSignals
from ..robot_controller import RobotController

class FollowerPanel(QWidget):
    """跟随模式控制面板"""

    def __init__(self, controller: RobotController, signals: WorkerSignals, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.signals = signals
        
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        follower_group = QGroupBox("跟随模式控制")
        follower_layout = QGridLayout(follower_group)

        # 第一行：独立命令按钮
        self.start_follower_btn = QPushButton("start_follower")
        self.start_follower_btn.setEnabled(False)
        self.start_follower_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.start_follower_btn.clicked.connect(self.on_cmd_start_follower)
        follower_layout.addWidget(self.start_follower_btn, 0, 0)

        self.set_jog_coord_btn = QPushButton("set_jog_coordinate --tool")
        self.set_jog_coord_btn.setEnabled(False)
        self.set_jog_coord_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.set_jog_coord_btn.clicked.connect(self.on_cmd_set_jog_coordinate)
        follower_layout.addWidget(self.set_jog_coord_btn, 0, 1)

        self.follower_cart_btn = QPushButton("follower_cart")
        self.follower_cart_btn.setEnabled(False)
        self.follower_cart_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.follower_cart_btn.clicked.connect(self.on_cmd_follower_cart)
        follower_layout.addWidget(self.follower_cart_btn, 0, 2)

        # 第二行：停止跟随和一键启动
        self.stop_follower_btn = QPushButton("stop_follower")
        self.stop_follower_btn.setEnabled(False)
        self.stop_follower_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.stop_follower_btn.clicked.connect(self.on_cmd_stop_follower)
        follower_layout.addWidget(self.stop_follower_btn, 1, 0)

        self.one_click_follower_btn = QPushButton("一键启动跟随模式")
        self.one_click_follower_btn.setEnabled(False)
        self.one_click_follower_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.one_click_follower_btn.clicked.connect(self.on_one_click_follower)
        follower_layout.addWidget(self.one_click_follower_btn, 1, 1, 1, 2)

        # 坐标系显示
        coord_frame = QFrame()
        coord_frame_layout = QHBoxLayout(coord_frame)
        coord_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.coord_label = QLabel("当前坐标系:")
        coord_frame_layout.addWidget(self.coord_label)
        self.coord_value = QLabel("Joint")
        self.coord_value.setStyleSheet("color: #feca57; font-weight: bold;")
        coord_frame_layout.addWidget(self.coord_value)
        coord_frame_layout.addStretch()
        follower_layout.addWidget(coord_frame, 2, 0, 1, 3)

        layout.addWidget(follower_group)

    def connect_signals(self):
        """连接信号"""
        self.signals.status_updated.connect(self._on_status_updated)
        self.signals.command_finished.connect(self._on_command_finished)

    def on_cmd_start_follower(self):
        """start_follower 命令"""
        self.signals.status_updated.emit("执行 start_follower...")
        threading.Thread(target=self.controller.cmd_start_follower, daemon=True).start()

    def on_cmd_set_jog_coordinate(self):
        """set_jog_coordinate --tool 命令"""
        self.signals.status_updated.emit("执行 set_jog_coordinate --tool...")
        def task():
            if self.controller.cmd_set_jog_coordinate_tool():
                self.coord_value.setText("Tool")
        threading.Thread(target=task, daemon=True).start()

    def on_cmd_follower_cart(self):
        """follower_cart 命令"""
        self.signals.status_updated.emit("执行 follower_cart...")
        def task():
            if self.controller.cmd_follower_cart():
                self.signals.status_updated.emit("follower_cart 已启动")
        threading.Thread(target=task, daemon=True).start()

    def on_cmd_stop_follower(self):
        """stop_follower 命令"""
        self.signals.status_updated.emit("执行 stop_follower...")
        def task():
            if self.controller.cmd_stop_follower():
                self.signals.status_updated.emit("跟随模式已停止")
        threading.Thread(target=task, daemon=True).start()

    def on_one_click_follower(self, ip=None):
        """一键启动跟随模式"""
        # 注意：这里需要 IP，如果是从主窗口调用，需要传入 IP
        if ip is None:
            # 尝试从 parent 或其他地方获取，或者由 main_window 触发并传参
            return
            
        self.signals.status_updated.emit("正在启动跟随模式...")
        self.one_click_follower_btn.setEnabled(False)

        def task():
            success = self.controller.start_follower_mode(ip)
            if success:
                self.signals.command_finished.emit(True, "跟随模式已启动")
            else:
                self.signals.command_finished.emit(False, "跟随模式启动失败")
        threading.Thread(target=task, daemon=True).start()

    def _on_status_updated(self, message):
        """状态更新 UI 反馈"""
        if "使能成功" in message:
            self.set_controls_enabled(True)

        if "已退出登录" in message:
            self.set_controls_enabled(False)

        if "跟随模式已启动" in message or "follower_cart 已启动" in message:
            self.one_click_follower_btn.setEnabled(True)
            self.coord_value.setText("Tool")

        if "跟随模式已停止" in message:
            self.one_click_follower_btn.setEnabled(True)
            self.coord_value.setText("Joint")

    def _on_command_finished(self, success, message):
        """命令执行完成"""
        if "跟随模式" in message:
            self.one_click_follower_btn.setEnabled(True)

    def set_controls_enabled(self, enabled):
        """启用/禁用所有跟随控制按钮"""
        self.start_follower_btn.setEnabled(enabled)
        self.set_jog_coord_btn.setEnabled(enabled)
        self.follower_cart_btn.setEnabled(enabled)
        self.stop_follower_btn.setEnabled(enabled)
        self.one_click_follower_btn.setEnabled(enabled)
