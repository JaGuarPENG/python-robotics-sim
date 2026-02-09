"""
机器人状态和 TCP 显示面板
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from ..signals import WorkerSignals
from ..robot_controller import RobotController

class RobotStatusPanel(QWidget):
    """机器人状态和 TCP 显示面板"""
    
    # 定义切换模型信号
    model_toggle_requested = pyqtSignal()

    def __init__(self, controller: RobotController, signals: WorkerSignals, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.signals = signals
        self.tcp_labels = {}
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. 末端位置显示 (TCP)
        tcp_group = QGroupBox("末端位置 (TCP)")
        tcp_layout = QGridLayout(tcp_group)

        for i, (name, unit) in enumerate([('X', 'mm'), ('Y', 'mm'), ('Z', 'mm'),
                                           ('Rx', '°'), ('Ry', '°'), ('Rz', '°')]):
            label = QLabel(f"{name}:")
            label.setFont(QFont("Arial", 10, QFont.Bold))
            tcp_layout.addWidget(label, i // 3, (i % 3) * 2)

            value = QLabel("0.00")
            value.setFont(QFont("Courier New", 11))
            value.setStyleSheet("color: #00ffff;")
            tcp_layout.addWidget(value, i // 3, (i % 3) * 2 + 1)
            self.tcp_labels[name] = value

        note_label = QLabel("* 号表示模型计算值，可能与实际有偏差")
        note_label.setStyleSheet("color: #888; font-size: 9px;")
        tcp_layout.addWidget(note_label, 2, 0, 1, 6)
        layout.addWidget(tcp_group)

        # 2. 机器人状态显示
        status_group = QGroupBox("机器人状态")
        status_layout = QGridLayout(status_group)

        status_layout.addWidget(QLabel("运行状态:"), 0, 0)
        self.robot_status_value = QLabel("--")
        self.robot_status_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_status_value, 0, 1)

        status_layout.addWidget(QLabel("激活状态:"), 0, 2)
        self.robot_activate_value = QLabel("--")
        self.robot_activate_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_activate_value, 0, 3)

        status_layout.addWidget(QLabel("运动状态:"), 1, 0)
        self.robot_motion_value = QLabel("--")
        self.robot_motion_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_motion_value, 1, 1)

        status_layout.addWidget(QLabel("模式:"), 1, 2)
        self.robot_mode_value = QLabel("--")
        self.robot_mode_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_mode_value, 1, 3)

        self.refresh_status_btn = QPushButton("刷新状态")
        self.refresh_status_btn.setEnabled(False)
        self.refresh_status_btn.clicked.connect(self.on_refresh_status)
        status_layout.addWidget(self.refresh_status_btn, 2, 0, 1, 1)

        self.toggle_model_btn = QPushButton("切换模型 (3D)")
        self.toggle_model_btn.setStyleSheet("background-color: #34495e; color: white;")
        self.toggle_model_btn.clicked.connect(self.model_toggle_requested.emit)
        status_layout.addWidget(self.toggle_model_btn, 2, 1, 1, 1)

        status_layout.addWidget(QLabel("错误信息:"), 2, 2)
        self.robot_error_value = QLabel("无")
        self.robot_error_value.setStyleSheet("color: #2ecc71;")
        status_layout.addWidget(self.robot_error_value, 2, 3)

        layout.addWidget(status_group)

    def connect_signals(self):
        """连接信号"""
        self.signals.robot_status_updated.connect(self.update_robot_status_display)

    def on_refresh_status(self):
        """刷新状态按钮点击"""
        self.signals.status_updated.emit("正在查询状态...")
        threading.Thread(target=self.controller.get_robot_status, daemon=True).start()

    def update_robot_status_display(self, status_info):
        """更新状态显示"""
        # 运行状态
        status = status_info.get('status', '--')
        self.robot_status_value.setText(status)
        if status == 'Error':
            self.robot_status_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        elif status == 'Running':
            self.robot_status_value.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self.robot_status_value.setStyleSheet("color: #feca57; font-weight: bold;")

        # 激活状态
        activate = status_info.get('activate', '--')
        self.robot_activate_value.setText(activate)
        if activate in ['Enabled', 'Active']:
            self.robot_activate_value.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self.robot_activate_value.setStyleSheet("color: #feca57; font-weight: bold;")

        # 运动状态
        motion = status_info.get('motion', '--')
        self.robot_motion_value.setText(motion)
        if motion == 'Running':
            self.robot_motion_value.setStyleSheet("color: #2ecc71; font-weight: bold;")
        elif motion == 'Stop':
            self.robot_motion_value.setStyleSheet("color: #feca57; font-weight: bold;")
        else:
            self.robot_motion_value.setStyleSheet("color: #feca57; font-weight: bold;")

        # 模式
        mode = status_info.get('mode', '--')
        self.robot_mode_value.setText(mode)

        # 错误信息
        error = status_info.get('error', '无')
        self.robot_error_value.setText(error)
        if error == '无':
            self.robot_error_value.setStyleSheet("color: #2ecc71;")
        else:
            self.robot_error_value.setStyleSheet("color: #e74c3c;")

    def update_tcp_display(self, actual_tcp=None, model_pos=None, model_rpy=None):
        """更新 TCP 显示"""
        if actual_tcp and len(actual_tcp) >= 6:
            self.tcp_labels['X'].setText(f"{actual_tcp[0]:.2f}")
            self.tcp_labels['Y'].setText(f"{actual_tcp[1]:.2f}")
            self.tcp_labels['Z'].setText(f"{actual_tcp[2]:.2f}")
            self.tcp_labels['Rx'].setText(f"{actual_tcp[3]:.2f}")
            self.tcp_labels['Ry'].setText(f"{actual_tcp[4]:.2f}")
            self.tcp_labels['Rz'].setText(f"{actual_tcp[5]:.2f}")
        elif model_pos is not None and model_rpy is not None:
            self.tcp_labels['X'].setText(f"{model_pos[0]:.2f}*")
            self.tcp_labels['Y'].setText(f"{model_pos[1]:.2f}*")
            self.tcp_labels['Z'].setText(f"{model_pos[2]:.2f}*")
            self.tcp_labels['Rx'].setText(f"{model_rpy[0]:.2f}*")
            self.tcp_labels['Ry'].setText(f"{model_rpy[1]:.2f}*")
            self.tcp_labels['Rz'].setText(f"{model_rpy[2]:.2f}*")

    def set_refresh_enabled(self, enabled):
        """启用/禁用刷新按钮"""
        self.refresh_status_btn.setEnabled(enabled)
