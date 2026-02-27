"""
连接控制面板组件
"""

import threading
from PyQt5.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QLineEdit, QPushButton, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt

from ..config import ROBOT_IP, PORT_CONTROL
from ..signals import WorkerSignals
from ..robot_controller import RobotController

class ConnectionPanel(QGroupBox):
    """连接控制面板"""

    def __init__(self, controller: RobotController, signals: WorkerSignals, parent=None):
        super().__init__("连接控制", parent)
        self.controller = controller
        self.signals = signals
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """初始化 UI"""
        layout = QGridLayout(self)

        layout.addWidget(QLabel("IP:"), 0, 0)
        self.ip_input = QLineEdit(ROBOT_IP)
        self.ip_input.setMinimumWidth(160)
        self.ip_input.setMaximumWidth(220)
        self.ip_input.setStyleSheet("color: #f1c40f; font-weight: bold; padding: 2px;")
        layout.addWidget(self.ip_input, 0, 1)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.on_connect)
        layout.addWidget(self.connect_btn, 0, 2, 1, 2)

        self.login_btn = QPushButton("登录")
        self.login_btn.setEnabled(False)
        self.login_btn.clicked.connect(self.on_login)
        layout.addWidget(self.login_btn, 1, 0)

        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.setEnabled(False)
        self.logout_btn.clicked.connect(self.on_logout)
        layout.addWidget(self.logout_btn, 1, 1)

        self.enable_btn = QPushButton("使能")
        self.enable_btn.setEnabled(False)
        self.enable_btn.clicked.connect(self.on_enable)
        layout.addWidget(self.enable_btn, 1, 2)

        # 状态指示
        status_frame = QFrame()
        status_frame_layout = QGridLayout(status_frame)
        status_frame_layout.setContentsMargins(5, 5, 5, 5)

        self.ws_status = QLabel("●")
        self.ws_status.setStyleSheet("color: red; font-size: 16px;")
        status_frame_layout.addWidget(QLabel("连接:"), 0, 0)
        status_frame_layout.addWidget(self.ws_status, 0, 1)

        self.login_status = QLabel("●")
        self.login_status.setStyleSheet("color: gray; font-size: 16px;")
        status_frame_layout.addWidget(QLabel("登录:"), 0, 2)
        status_frame_layout.addWidget(self.login_status, 0, 3)

        self.enable_status = QLabel("●")
        self.enable_status.setStyleSheet("color: gray; font-size: 16px;")
        status_frame_layout.addWidget(QLabel("使能:"), 1, 0)
        status_frame_layout.addWidget(self.enable_status, 1, 1)

        self.follower_status = QLabel("●")
        self.follower_status.setStyleSheet("color: gray; font-size: 16px;")
        status_frame_layout.addWidget(QLabel("跟随:"), 1, 2)
        status_frame_layout.addWidget(self.follower_status, 1, 3)

        layout.addWidget(status_frame, 0, 4, 2, 1)

    def connect_signals(self):
        """连接信号"""
        self.signals.connection_changed.connect(self._on_connection_changed)
        self.signals.status_updated.connect(self._on_status_updated)
        self.signals.error_occurred.connect(self._on_error)

    def on_connect(self):
        """连接按钮点击"""
        ip = self.ip_input.text().strip()
        if not ip:
            QMessageBox.warning(self, "警告", "请输入IP地址")
            return

        self.connect_btn.setEnabled(False)
        # 通知主窗口或状态栏（通过信号）
        self.signals.status_updated.emit("正在连接...")

        def connect_task():
            success = self.controller.connect(ip, PORT_CONTROL)
            if success:
                self.controller.start_monitoring()
            else:
                # 如果连接失败，重新启用连接按钮
                # 注意：这应该在主线程做，但 signals 已经发了 error，_on_error 会处理
                pass

        threading.Thread(target=connect_task, daemon=True).start()

    def on_login(self):
        """登录按钮点击"""
        self.signals.status_updated.emit("正在登录...")
        threading.Thread(target=self.controller.login, daemon=True).start()

    def on_logout(self):
        """退出登录按钮点击"""
        self.signals.status_updated.emit("正在退出登录...")
        threading.Thread(target=self.controller.logout, daemon=True).start()

    def on_enable(self):
        """使能按钮点击"""
        self.signals.status_updated.emit("正在使能...")
        threading.Thread(target=self.controller.enable, daemon=True).start()

    def _on_connection_changed(self, connected, port_type):
        """连接状态变化 UI 更新"""
        if port_type == "websocket":
            color = "lime" if connected else "red"
            self.ws_status.setStyleSheet(f"color: {color}; font-size: 16px;")
            if connected:
                self.login_btn.setEnabled(True)
            self.connect_btn.setEnabled(True)

    def _on_status_updated(self, message):
        """状态更新 UI 反馈"""
        if "登录成功" in message:
            self.login_status.setStyleSheet("color: lime; font-size: 16px;")
            self.enable_btn.setEnabled(True)
            self.logout_btn.setEnabled(True)

        if "已退出登录" in message:
            self.login_status.setStyleSheet("color: gray; font-size: 16px;")
            self.enable_status.setStyleSheet("color: gray; font-size: 16px;")
            self.follower_status.setStyleSheet("color: gray; font-size: 16px;")
            self.enable_btn.setEnabled(False)
            self.logout_btn.setEnabled(False)

        if "使能成功" in message:
            self.enable_status.setStyleSheet("color: lime; font-size: 16px;")

        if "跟随模式已启动" in message:
            self.follower_status.setStyleSheet("color: lime; font-size: 16px;")
        elif "跟随模式已停止" in message:
            self.follower_status.setStyleSheet("color: gray; font-size: 16px;")

    def _on_error(self, error_msg):
        """错误发生时的 UI 处理"""
        self.connect_btn.setEnabled(True)

    def get_ip(self):
        """获取输入的 IP 地址"""
        return self.ip_input.text().strip()
