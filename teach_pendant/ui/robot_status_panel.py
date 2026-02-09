"""
机器人状态和 TCP 显示面板 (布局优化版：误差显示在按钮位置)
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from ..signals import WorkerSignals
from ..robot_controller import RobotController

class RobotStatusPanel(QWidget):
    """机器人状态和 TCP 显示面板"""
    
    # 定义切换模型信号 (保留接口兼容)
    model_toggle_requested = pyqtSignal()

    def __init__(self, controller: RobotController, signals: WorkerSignals, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.signals = signals
        self.tcp_labels = {}
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """初始化 UI (针对全屏大字号优化)"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. 末端位置显示 (TCP)
        tcp_group = QGroupBox("末端位姿 (Real-time TCP)")
        tcp_layout = QGridLayout(tcp_group)

        for i, (name, unit) in enumerate([('X', 'mm'), ('Y', 'mm'), ('Z', 'mm'),
                                           ('Rx', '°'), ('Ry', '°'), ('Rz', '°')]):
            label = QLabel(f"{name}:")
            label.setFont(QFont("Arial", 18, QFont.Bold))
            tcp_layout.addWidget(label, i // 3, (i % 3) * 2)

            value = QLabel("0.00")
            value.setFont(QFont("Courier New", 30, QFont.Bold))
            value.setStyleSheet("color: #00ffff;")
            tcp_layout.addWidget(value, i // 3, (i % 3) * 2 + 1)
            self.tcp_labels[name] = value

        note_label = QLabel("* 表示模型计算值")
        note_label.setStyleSheet("color: #888; font-size: 14px;")
        tcp_layout.addWidget(note_label, 2, 0, 1, 6)
        layout.addWidget(tcp_group)

        # 2. 机器人状态与误差 (合并显示)
        status_group = QGroupBox("系统状态与跟踪误差")
        status_layout = QGridLayout(status_group)

        # Row 0
        status_layout.addWidget(QLabel("运行状态:"), 0, 0)
        self.robot_status_value = QLabel("--")
        self.robot_status_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_status_value, 0, 1)

        status_layout.addWidget(QLabel("激活状态:"), 0, 2)
        self.robot_activate_value = QLabel("--")
        self.robot_activate_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_activate_value, 0, 3)

        # Row 1
        status_layout.addWidget(QLabel("运动状态:"), 1, 0)
        self.robot_motion_value = QLabel("--")
        self.robot_motion_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_motion_value, 1, 1)

        status_layout.addWidget(QLabel("控制模式:"), 1, 2)
        self.robot_mode_value = QLabel("--")
        self.robot_mode_value.setStyleSheet("color: #feca57; font-weight: bold;")
        status_layout.addWidget(self.robot_mode_value, 1, 3)

        # Row 2: 原本按钮的位置，现在显示误差 (核心位置)
        err_title = QLabel("线性偏差:")
        err_title.setStyleSheet("font-weight: bold; color: #ecf0f1;")
        status_layout.addWidget(err_title, 2, 0)
        
        self.linear_error_label = QLabel("0.00 mm")
        self.linear_error_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 36px;") # 极大字号
        status_layout.addWidget(self.linear_error_label, 2, 1)

        # 报警信息
        status_layout.addWidget(QLabel("报警:"), 2, 2)
        self.robot_error_value = QLabel("无")
        self.robot_error_value.setStyleSheet("color: #2ecc71;")
        status_layout.addWidget(self.robot_error_value, 2, 3)

        # 刷新按钮 (放在最后，不占用主要网格)
        btn_layout = QHBoxLayout()
        self.refresh_status_btn = QPushButton("手动刷新状态")
        self.refresh_status_btn.setMinimumHeight(50)
        self.refresh_status_btn.setEnabled(False)
        self.refresh_status_btn.clicked.connect(self.on_refresh_status)
        btn_layout.addWidget(self.refresh_status_btn)
        
        layout.addLayout(status_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(status_group)

    def connect_signals(self):
        """连接信号"""
        self.signals.robot_status_updated.connect(self.update_robot_status_display)
        self.signals.tracking_error_updated.connect(self.update_tracking_error)

    def update_robot_status_display(self, status_info):
        """更新状态显示"""
        status = status_info.get('status', '--')
        self.robot_status_value.setText(status)
        self.robot_status_value.setStyleSheet(f"color: {'#e74c3c' if status == 'Error' else '#2ecc71' if status == 'Running' else '#feca57'}; font-weight: bold;")

        activate = status_info.get('activate', '--')
        self.robot_activate_value.setText(activate)
        self.robot_activate_value.setStyleSheet(f"color: {'#2ecc71' if activate in ['Enabled', 'Active'] else '#feca57'}; font-weight: bold;")

        motion = status_info.get('motion', '--')
        self.robot_motion_value.setText(motion)
        self.robot_motion_value.setStyleSheet(f"color: {'#2ecc71' if motion == 'Running' else '#feca57'}; font-weight: bold;")

        self.robot_mode_value.setText(status_info.get('mode', '--'))

        error = status_info.get('error', '无')
        self.robot_error_value.setText(error)
        self.robot_error_value.setStyleSheet(f"color: {'#2ecc71' if error == '无' else '#e74c3c'};")

    def update_tracking_error(self, linear_err, angular_err):
        """更新偏差数值"""
        self.linear_error_label.setText(f"{linear_err:.2f} mm")

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
        self.refresh_status_btn.setEnabled(enabled)

    def on_refresh_status(self):
        self.signals.status_updated.emit("正在查询状态...")
        threading.Thread(target=self.controller.get_robot_status, daemon=True).start()