"""
关节控制面板组件 (Joint, Jog, Presets, Velocity)
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ..config import (
    PRESET_POSITIONS, JOG_STEP_SMALL, JOG_STEP_MEDIUM, JOG_STEP_LARGE
)
from ..signals import WorkerSignals
from ..robot_controller import RobotController

class JointControlPanel(QWidget):
    """关节控制面板"""

    def __init__(self, controller: RobotController, signals: WorkerSignals, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.signals = signals
        self.jog_step = JOG_STEP_MEDIUM
        
        self.joint_values = []
        self.jog_minus_btns = []
        self.jog_plus_btns = []
        self.preset_btns = []
        
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. 关节位置显示
        joint_group = QGroupBox("关节位置 (度)")
        joint_layout = QGridLayout(joint_group)
        for i in range(6):
            label = QLabel(f"J{i+1}:")
            label.setFont(QFont("Arial", 11, QFont.Bold))
            joint_layout.addWidget(label, i // 2, (i % 2) * 2)

            value = QLabel("0.000")
            value.setFont(QFont("Courier New", 12))
            value.setStyleSheet("background-color: #1a1a2e; color: #00ff00; padding: 3px; border-radius: 3px;")
            value.setMinimumWidth(80)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            joint_layout.addWidget(value, i // 2, (i % 2) * 2 + 1)
            self.joint_values.append(value)
        layout.addWidget(joint_group)

        # 2. 关节控制 (Jog)
        jog_group = QGroupBox("关节控制 (Jog)")
        jog_layout = QGridLayout(jog_group)
        
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("步进:"))
        self.step_small_btn = QPushButton(f"{JOG_STEP_SMALL}°")
        self.step_small_btn.setCheckable(True)
        self.step_small_btn.setEnabled(False)
        self.step_small_btn.clicked.connect(lambda: self.set_jog_step(JOG_STEP_SMALL))
        step_layout.addWidget(self.step_small_btn)

        self.step_medium_btn = QPushButton(f"{JOG_STEP_MEDIUM}°")
        self.step_medium_btn.setCheckable(True)
        self.step_medium_btn.setChecked(True)
        self.step_medium_btn.setEnabled(False)
        self.step_medium_btn.clicked.connect(lambda: self.set_jog_step(JOG_STEP_MEDIUM))
        step_layout.addWidget(self.step_medium_btn)

        self.step_large_btn = QPushButton(f"{JOG_STEP_LARGE}°")
        self.step_large_btn.setCheckable(True)
        self.step_large_btn.setEnabled(False)
        self.step_large_btn.clicked.connect(lambda: self.set_jog_step(JOG_STEP_LARGE))
        step_layout.addWidget(self.step_large_btn)
        jog_layout.addLayout(step_layout, 0, 0, 1, 4)

        for i in range(6):
            row = i // 2 + 1
            col_offset = (i % 2) * 2
            label = QLabel(f"J{i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 10, QFont.Bold))
            jog_layout.addWidget(label, row, col_offset)

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setSpacing(2)

            minus_btn = QPushButton("-")
            minus_btn.setFixedSize(35, 30)
            minus_btn.setFont(QFont("Arial", 12, QFont.Bold))
            minus_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
            minus_btn.setEnabled(False)
            minus_btn.clicked.connect(lambda checked, idx=i: self.jog_joint(idx, -1))
            btn_layout.addWidget(minus_btn)
            self.jog_minus_btns.append(minus_btn)

            plus_btn = QPushButton("+")
            plus_btn.setFixedSize(35, 30)
            plus_btn.setFont(QFont("Arial", 12, QFont.Bold))
            plus_btn.setStyleSheet("background-color: #51cf66; color: white;")
            plus_btn.setEnabled(False)
            plus_btn.clicked.connect(lambda checked, idx=i: self.jog_joint(idx, 1))
            btn_layout.addWidget(plus_btn)
            self.jog_plus_btns.append(plus_btn)
            jog_layout.addWidget(btn_widget, row, col_offset + 1)
        layout.addWidget(jog_group)

        # 3. 预设位置
        preset_group = QGroupBox("预设位置")
        preset_layout = QGridLayout(preset_group)
        row, col = 0, 0
        for name, joints in PRESET_POSITIONS.items():
            btn = QPushButton(name)
            btn.setMinimumHeight(40)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, n=name, j=joints: self.on_preset_clicked(n, j))
            preset_layout.addWidget(btn, row, col)
            self.preset_btns.append(btn)
            col += 1
            if col > 1: col = 0; row += 1
        layout.addWidget(preset_group)

        # 4. 速度控制
        vel_group = QGroupBox("速度控制")
        vel_layout = QVBoxLayout(vel_group)
        vel_label_layout = QHBoxLayout()
        vel_label_layout.addWidget(QLabel("运动速度:"))
        self.vel_value_label = QLabel("100%")
        self.vel_value_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.vel_value_label.setStyleSheet("color: #feca57;")
        vel_label_layout.addWidget(self.vel_value_label)
        vel_label_layout.addStretch()
        vel_layout.addLayout(vel_label_layout)

        self.vel_slider = QSlider(Qt.Horizontal)
        self.vel_slider.setMinimum(1)
        self.vel_slider.setMaximum(100)
        self.vel_slider.setValue(100)
        self.vel_slider.setEnabled(False)
        self.vel_slider.valueChanged.connect(self.on_velocity_slider_changed)
        vel_layout.addWidget(self.vel_slider)

        vel_btn_layout = QHBoxLayout()
        for pct in [10, 25, 50, 75, 100]:
            btn = QPushButton(f"{pct}%")
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, v=pct: self.set_velocity(v))
            vel_btn_layout.addWidget(btn)
        vel_layout.addLayout(vel_btn_layout)
        layout.addWidget(vel_group)

    def on_preset_clicked(self, name, target_joints):
        """点击预设位置：确认并异步执行"""
        msg = f"是否立即移动到预设位置 [{name}]?\n\n目标角度:\n{['%.2f'%j for j in target_joints]}"
        reply = QMessageBox.question(self, "确认运动", msg, QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.signals.status_updated.emit(f"正在前往预设位置: {name}")
            # 使用安全速度
            vels = [20, 20, 20] 
            threading.Thread(target=lambda: self.controller.move_joint(target_joints, vels=vels), daemon=True).start()

    def connect_signals(self):
        """连接信号"""
        self.signals.joints_updated.connect(self.update_joint_display)

    def update_joint_display(self, joints):
        """更新关节位置显示"""
        for i, val in enumerate(joints[:6]):
            if i < len(self.joint_values):
                self.joint_values[i].setText(f"{val:.3f}")

    def set_jog_step(self, step):
        """设置 Jog 步进"""
        self.jog_step = step
        self.step_small_btn.setChecked(step == JOG_STEP_SMALL)
        self.step_medium_btn.setChecked(step == JOG_STEP_MEDIUM)
        self.step_large_btn.setChecked(step == JOG_STEP_LARGE)

    def jog_joint(self, joint_index, direction):
        """执行 Jog"""
        self.controller.jog_joint(joint_index, direction, self.jog_step)

    def on_velocity_slider_changed(self, value):
        """速度滑块变化"""
        self.vel_value_label.setText(f"{value}%")

    def set_velocity(self, percent):
        """设置速度并更新 UI"""
        self.vel_slider.setValue(percent)
        self.controller.set_velocity(percent)

    def enable_controls(self, enabled):
        """启用/禁用所有运动控制"""
        for btn in self.jog_minus_btns + self.jog_plus_btns + self.preset_btns:
            btn.setEnabled(enabled)
        
        self.step_small_btn.setEnabled(enabled)
        self.step_medium_btn.setEnabled(enabled)
        self.step_large_btn.setEnabled(enabled)
        self.vel_slider.setEnabled(enabled)