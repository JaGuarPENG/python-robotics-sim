"""
关节控制面板组件 (全屏、大字号版 - 恢复 +/- 样式与颜色)
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ..config import (
    PRESET_POSITIONS, JOG_STEP_SMALL, JOG_STEP_MEDIUM, JOG_STEP_LARGE
)
from ..signals import WorkerSignals
from ..robot_controller import RobotController

class JointControlPanel(QWidget):
    """关节控制面板 - 针对工业触摸屏优化"""

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
        """初始化 UI (工业大字号样式)"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 1. 关节角度显示
        joint_group = QGroupBox("关节角度 (Joint Angles)")
        joint_layout = QGridLayout(joint_group)
        for i in range(6):
            label = QLabel(f"J{i+1}:")
            label.setFont(QFont("Arial", 18, QFont.Bold))
            joint_layout.addWidget(label, i // 2, (i % 2) * 2)

            value = QLabel("0.000")
            value.setFont(QFont("Courier New", 24, QFont.Bold))
            value.setStyleSheet("background-color: #1a1a2e; color: #00ff00; padding: 5px; border-radius: 5px;")
            value.setMinimumWidth(130)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            joint_layout.addWidget(value, i // 2, (i % 2) * 2 + 1)
            self.joint_values.append(value)
        layout.addWidget(joint_group)

        # 2. 手动点动控制
        jog_group = QGroupBox("手动点动 (Joint Jog)")
        jog_layout = QGridLayout(jog_group)
        
        step_layout = QHBoxLayout()
        step_label = QLabel("步进:")
        step_label.setFont(QFont("Arial", 16))
        step_layout.addWidget(step_label)
        
        self.step_btns = []
        for step in [JOG_STEP_SMALL, JOG_STEP_MEDIUM, JOG_STEP_LARGE]:
            btn = QPushButton(f"{step}°")
            btn.setCheckable(True)
            btn.setMinimumHeight(55)
            btn.setEnabled(False)
            if step == JOG_STEP_MEDIUM: btn.setChecked(True)
            btn.clicked.connect(lambda checked, s=step: self.set_jog_step(s))
            step_layout.addWidget(btn)
            self.step_btns.append(btn)
            if step == JOG_STEP_SMALL: self.step_small_btn = btn
            if step == JOG_STEP_MEDIUM: self.step_medium_btn = btn
            if step == JOG_STEP_LARGE: self.step_large_btn = btn
            
        jog_layout.addLayout(step_layout, 0, 0, 1, 4)

        for i in range(6):
            row = i // 2 + 1
            col_offset = (i % 2) * 2
            label = QLabel(f"J{i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 18, QFont.Bold))
            jog_layout.addWidget(label, row, col_offset)

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setSpacing(8)

            minus_btn = QPushButton("-")
            minus_btn.setMinimumSize(75, 65) # 大尺寸
            minus_btn.setFont(QFont("Arial", 26, QFont.Bold))
            minus_btn.setStyleSheet("background-color: #e74c3c; color: white; border-radius: 8px;")
            minus_btn.setEnabled(False)
            minus_btn.clicked.connect(lambda checked, idx=i: self.jog_joint(idx, -1))
            btn_layout.addWidget(minus_btn)
            self.jog_minus_btns.append(minus_btn)

            plus_btn = QPushButton("+")
            plus_btn.setMinimumSize(75, 65) # 大尺寸
            plus_btn.setFont(QFont("Arial", 26, QFont.Bold))
            plus_btn.setStyleSheet("background-color: #2ecc71; color: white; border-radius: 8px;")
            plus_btn.setEnabled(False)
            plus_btn.clicked.connect(lambda checked, idx=i: self.jog_joint(idx, 1))
            btn_layout.addWidget(plus_btn)
            self.jog_plus_btns.append(plus_btn)
            jog_layout.addWidget(btn_widget, row, col_offset + 1)
        layout.addWidget(jog_group)

        # 3. 快速定位
        preset_group = QGroupBox("快速定位 (Presets)")
        preset_layout = QGridLayout(preset_group)
        row, col = 0, 0
        for name, joints in PRESET_POSITIONS.items():
            btn = QPushButton(name)
            btn.setMinimumHeight(65)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, n=name, j=joints: self.on_preset_clicked(n, j))
            preset_layout.addWidget(btn, row, col)
            self.preset_btns.append(btn)
            col += 1
            if col > 1: col = 0; row += 1
        layout.addWidget(preset_group)

        layout.addStretch()

    def set_jog_step(self, step):
        self.jog_step = step
        self.step_small_btn.setChecked(step == JOG_STEP_SMALL)
        self.step_medium_btn.setChecked(step == JOG_STEP_MEDIUM)
        self.step_large_btn.setChecked(step == JOG_STEP_LARGE)

    def jog_joint(self, joint_index, direction):
        self.controller.jog_joint(joint_index, direction, self.jog_step)

    def on_preset_clicked(self, name, target_joints):
        msg = f"确定移动到 [{name}] 吗？"
        if QMessageBox.question(self, "确认", msg) == QMessageBox.Yes:
            threading.Thread(target=lambda: self.controller.move_joint(target_joints, vels=[20,20,20]), daemon=True).start()

    def connect_signals(self):
        self.signals.joints_updated.connect(self.update_joint_display)

    def update_joint_display(self, joints):
        for i, val in enumerate(joints[:6]):
            if i < len(self.joint_values):
                self.joint_values[i].setText(f"{val:.3f}")

    def enable_controls(self, enabled):
        for btn in self.jog_minus_btns + self.jog_plus_btns + self.preset_btns + self.step_btns:
            btn.setEnabled(enabled)
