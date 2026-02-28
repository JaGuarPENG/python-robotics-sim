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
        """初始化 UI (均衡美化版)"""
        # 统一字体大小，确保清晰度与平衡感
        self.setStyleSheet("""
            QLabel { font-size: 30px; } 
            QGroupBox { font-size: 35px; font-weight: bold; margin-top: 15px; padding-top: 20px; }
            QPushButton#small_refresh_btn { 
                font-size: 30px; 
                min-height: 45px; 
                min-width: 100px;
                border: 2px solid #555; 
                border-radius: 8px;
                background-color: #34495e;
            }
            QPushButton#small_refresh_btn:hover { background-color: #4a627a; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15) # 增加组间距，防止挤在一起

        # 1. 末端位置显示 (TCP)
        tcp_group = QGroupBox("末端位姿 (Real-time TCP)")
        tcp_layout = QGridLayout(tcp_group)
        tcp_layout.setSpacing(15) # 增加单元格间距
        
        # 设置 6 列等宽，确保布局整齐均衡
        for c in range(6):
            tcp_layout.setColumnStretch(c, 1)

        for i, (name, unit) in enumerate([('X', 'mm'), ('Y', 'mm'), ('Z', 'mm'),
                                           ('Rx', '°'), ('Ry', '°'), ('Rz', '°')]):
            label = QLabel(f"{name}:")
            label.setStyleSheet("font-weight: bold; color: #bdc3c7;")
            tcp_layout.addWidget(label, i // 3, (i % 3) * 2)

            value = QLabel("0.00")
            value.setFont(QFont("Courier New", 26, QFont.Bold)) 
            value.setStyleSheet("color: #00ffff;")
            tcp_layout.addWidget(value, i // 3, (i % 3) * 2 + 1)
            self.tcp_labels[name] = value

        layout.addWidget(tcp_group)

        # 2. 机器人状态与误差
        status_group = QGroupBox("系统状态与偏差")
        status_layout = QGridLayout(status_group)
        status_layout.setSpacing(15)
        # 设置 4 列等宽
        for c in range(4):
            status_layout.setColumnStretch(c, 1)

        val_style = "color: #feca57; font-weight: bold; font-size: 20px;"
        lbl_style = "font-weight: bold; color: #bdc3c7;"

        # Row 0
        status_layout.addWidget(QLabel("运行:", styleSheet=lbl_style), 0, 0)
        self.robot_status_value = QLabel("--")
        self.robot_status_value.setStyleSheet(val_style)
        status_layout.addWidget(self.robot_status_value, 0, 1)

        status_layout.addWidget(QLabel("激活:", styleSheet=lbl_style), 0, 2)
        self.robot_activate_value = QLabel("--")
        self.robot_activate_value.setStyleSheet(val_style)
        status_layout.addWidget(self.robot_activate_value, 0, 3)

        # Row 1
        status_layout.addWidget(QLabel("运动:", styleSheet=lbl_style), 1, 0)
        self.robot_motion_value = QLabel("--")
        self.robot_motion_value.setStyleSheet(val_style)
        status_layout.addWidget(self.robot_motion_value, 1, 1)

        status_layout.addWidget(QLabel("模式:", styleSheet=lbl_style), 1, 2)
        self.robot_mode_value = QLabel("--")
        self.robot_mode_value.setStyleSheet(val_style)
        status_layout.addWidget(self.robot_mode_value, 1, 3)

        # Row 2: 线性偏差与报警
        status_layout.addWidget(QLabel("偏差:", styleSheet=lbl_style), 2, 0)
        self.linear_error_label = QLabel("0.00 mm")
        self.linear_error_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 28px;") 
        status_layout.addWidget(self.linear_error_label, 2, 1)

        status_layout.addWidget(QLabel("报警:", styleSheet=lbl_style), 2, 2)
        self.robot_error_value = QLabel("无")
        self.robot_error_value.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 20px;")
        status_layout.addWidget(self.robot_error_value, 2, 3)

        # 刷新按钮 (独立一行，靠右)
        self.refresh_status_btn = QPushButton("刷新状态")
        self.refresh_status_btn.setObjectName("small_refresh_btn")
        self.refresh_status_btn.setEnabled(False)
        self.refresh_status_btn.clicked.connect(self.on_refresh_status)
        status_layout.addWidget(self.refresh_status_btn, 3, 3, Qt.AlignRight)
        
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