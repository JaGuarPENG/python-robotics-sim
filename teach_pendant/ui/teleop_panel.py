"""
遥操作测试面板 (UDP, Step-by-Step Test)
"""

import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton, QFrame
)
from PyQt5.QtCore import Qt

from ..signals import WorkerSignals
from ..robot_controller import RobotController

class TeleopPanel(QWidget):
    """遥操作测试面板"""

    def __init__(self, controller: RobotController, signals: WorkerSignals, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.signals = signals
        
        self.test_btns = []
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        teleop_group = QGroupBox("遥操作控制 (步进测试)")
        teleop_layout = QGridLayout(teleop_group)

        # UDP 连接控制
        self.udp_connect_btn = QPushButton("连接 UDP (9998)")
        self.udp_connect_btn.setEnabled(True)
        self.udp_connect_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.udp_connect_btn.clicked.connect(self.on_udp_connect)
        teleop_layout.addWidget(self.udp_connect_btn, 0, 0, 1, 2)

        self.udp_status = QLabel("●")
        self.udp_status.setStyleSheet("color: gray; font-size: 16px;")
        teleop_layout.addWidget(QLabel("UDP:"), 0, 2)
        teleop_layout.addWidget(self.udp_status, 0, 3)

        self.read_current_btn = QPushButton("获取当前基座坐标")
        self.read_current_btn.setEnabled(False)
        self.read_current_btn.clicked.connect(self.on_read_current_pose)
        teleop_layout.addWidget(self.read_current_btn, 1, 0, 1, 4)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        teleop_layout.addWidget(line, 2, 0, 1, 4)

        # 步进点位测试按钮
        test_btns_layout = QGridLayout()
        self.btn_x_plus = QPushButton("X +5")
        self.btn_y_plus = QPushButton("Y +5")
        self.btn_z_plus = QPushButton("Z +5")
        self.btn_x_minus = QPushButton("X -5")
        self.btn_y_minus = QPushButton("Y -5")
        self.btn_z_minus = QPushButton("Z -5")
        
        self.test_btns = [
            self.btn_x_plus, self.btn_y_plus, self.btn_z_plus,
            self.btn_x_minus, self.btn_y_minus, self.btn_z_minus
        ]
        
        for i, btn in enumerate(self.test_btns):
            btn.setEnabled(False)
            test_btns_layout.addWidget(btn, i // 3, i % 3)
            
        self.btn_x_plus.clicked.connect(lambda: self.on_step_test(5, 0, 0))
        self.btn_y_plus.clicked.connect(lambda: self.on_step_test(0, 5, 0))
        self.btn_z_plus.clicked.connect(lambda: self.on_step_test(0, 0, 5))
        self.btn_x_minus.clicked.connect(lambda: self.on_step_test(-5, 0, 0))
        self.btn_y_minus.clicked.connect(lambda: self.on_step_test(0, -5, 0))
        self.btn_z_minus.clicked.connect(lambda: self.on_step_test(0, 0, -5))

        teleop_layout.addLayout(test_btns_layout, 3, 0, 1, 4)

        self.reset_zero_btn = QPushButton("累积量归零 (Reset to Start)")
        self.reset_zero_btn.setEnabled(False)
        self.reset_zero_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; min-height: 35px;")
        self.reset_zero_btn.clicked.connect(self.on_reset_zero)
        teleop_layout.addWidget(self.reset_zero_btn, 4, 0, 1, 4)

        note = QLabel("提示: 按钮发送相对于启动点的累积增量")
        note.setStyleSheet("color: #888; font-size: 9px;")
        teleop_layout.addWidget(note, 5, 0, 1, 4)

        layout.addWidget(teleop_group)

    def connect_signals(self):
        """连接信号"""
        self.signals.status_updated.connect(self._on_status_updated)
        self.signals.command_finished.connect(lambda success, msg: self._on_status_updated(msg))

    def on_udp_connect(self):
        """UDP 连接/断开点击"""
        if self.controller.udp_connected:
            self.controller.disconnect_udp()
        else:
            self.signals.status_updated.emit("正在连接 UDP...")
            threading.Thread(target=lambda: self.controller.connect_udp(9998), daemon=True).start()

    def on_read_current_pose(self):
        """获取当前 TCP 位姿 (仅打印或记录)"""
        tcp = self.controller.get_current_tcp_mm_deg()
        if tcp:
            self.signals.status_updated.emit(f"当前 TCP: {['%.2f'%x for x in tcp]}")
        else:
            self.signals.status_updated.emit("无法获取当前 TCP")

    def on_step_test(self, dx, dy, dz):
        """步进增量测试"""
        self.signals.status_updated.emit(f"测试增量: ({dx}, {dy}, {dz})")
        threading.Thread(target=lambda: self.controller.send_raw_increment(dx, dy, dz), daemon=True).start()

    def on_reset_zero(self):
        """累积量归零"""
        def task():
            self.controller.follower_offset.fill(0)
            success = self.controller.udp_client.send_pose_euler(0, 0, 0, 0, 0, 0, "321")
            if success:
                self.signals.status_updated.emit("累积量已重置为 0")
        threading.Thread(target=task, daemon=True).start()

    def _on_status_updated(self, message):
        """状态更新 UI 反馈"""
        # 修正匹配字符串，使其与 RobotController 发出的消息一致
        if "UDP 连接成功" in message:
            self.udp_status.setStyleSheet("color: lime; font-size: 16px;")
            self.udp_connect_btn.setText("断开 UDP")
            self._update_test_btns()

        if "UDP 连接已断开" in message:
            self.udp_status.setStyleSheet("color: gray; font-size: 16px;")
            self.udp_connect_btn.setText("连接 UDP (9998)")
            self._update_test_btns()

        # 新增：当跟随模式状态变化时，也触发按钮更新
        if "跟随模式" in message or "follower_cart" in message:
            self._update_test_btns()

    def _update_test_btns(self):
        """根据跟随模式和 UDP 连接状态启用按钮"""
        can_test = self.controller.is_follower_mode and self.controller.udp_connected
        for btn in self.test_btns:
            btn.setEnabled(can_test)
        self.reset_zero_btn.setEnabled(can_test)

    def set_udp_enabled(self, enabled):
        """启用/禁用 UDP 连接按钮"""
        self.udp_connect_btn.setEnabled(enabled)
        self.read_current_btn.setEnabled(enabled)
        self._update_test_btns()
