"""
示教器主窗口
"""

import threading
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLabel, QSlider, QLineEdit,
    QMessageBox, QFrame, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from spatialmath import SE3 # 用于 IK 计算目标位姿

from .config import (
    ROBOT_IP, PORT_CONTROL, PRESET_POSITIONS,
    JOG_STEP_SMALL, JOG_STEP_MEDIUM, JOG_STEP_LARGE
)
from .signals import WorkerSignals
from .robot_3d_widget import Robot3DWidget
from .robot_controller import RobotController


class TeachPendantWindow(QMainWindow):
    """示教器主窗口"""

    # 原始增量测试 (dx, dy, dz, drx, dry, drz) - 单位: mm, 度
    # 直接发送固定增量，不计算目标-当前差值
    # 使用较大增量 (50mm) 测试，类似 C++ Demo 中的 100mm
    RAW_INCREMENT_TESTS = [
        (50, 0, 0, 0, 0, 0),    # 测试1: X +50mm
        (0, 50, 0, 0, 0, 0),    # 测试2: Y +50mm
        (0, 0, 50, 0, 0, 0),    # 测试3: Z +50mm
    ]

    # 保留原来的点位定义（只保留一个目标点）
    CONTINUOUS_TEST_POINTS = [
        (488, 164, 957, 225, -20, 0),
    ]

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.controller = RobotController(self.signals)
        self.jog_step = JOG_STEP_MEDIUM
        self.stop_continuous_flag = False  # 连续运动停止标志

        self.init_ui()
        self.connect_signals()

        # 启动 3D 视图更新定时器
        self.view_timer = QTimer()
        self.view_timer.timeout.connect(self.update_3d_view)
        self.view_timer.start(50)  # 20Hz 更新

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("机器人示教器 - Teach Pendant")
        self.setMinimumSize(1200, 800)

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ==========================================
        # 左侧：3D 可视化
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 3D 视图组
        view_group = QGroupBox("3D 机械臂视图")
        view_layout = QVBoxLayout(view_group)

        self.robot_view = Robot3DWidget()
        # 同步测试点位到 3D 视图
        self.robot_view.set_test_waypoints(self.CONTINUOUS_TEST_POINTS)
        view_layout.addWidget(self.robot_view)

        # 视图控制按钮
        view_btn_layout = QHBoxLayout()
        reset_view_btn = QPushButton("重置视角")
        reset_view_btn.clicked.connect(self.robot_view.reset_view)
        view_btn_layout.addWidget(reset_view_btn)
        
        # 新增：移动到目标点按钮
        self.move_target_btn = QPushButton("移动到红点 (IK)")
        self.move_target_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.move_target_btn.clicked.connect(self.on_move_to_target)
        view_btn_layout.addWidget(self.move_target_btn)
        
        # 新增：沿轨迹运动按钮
        self.run_traj_btn = QPushButton("画圆轨迹 (Trajectory)")
        self.run_traj_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")
        self.run_traj_btn.clicked.connect(self.on_run_trajectory)
        view_btn_layout.addWidget(self.run_traj_btn)
        
        view_btn_layout.addStretch()
        view_layout.addLayout(view_btn_layout)

        left_layout.addWidget(view_group)

        # 末端位置显示
        tcp_group = QGroupBox("末端位置 (TCP)")
        tcp_layout = QGridLayout(tcp_group)

        self.tcp_labels = {}
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

        # 说明标签
        note_label = QLabel("* 号表示模型计算值，可能与实际有偏差")
        note_label.setStyleSheet("color: #888; font-size: 9px;")
        tcp_layout.addWidget(note_label, 2, 0, 1, 6)

        left_layout.addWidget(tcp_group)

        # 机器人状态显示
        robot_status_group = QGroupBox("机器人状态")
        robot_status_layout = QGridLayout(robot_status_group)

        # 状态信息标签
        robot_status_layout.addWidget(QLabel("运行状态:"), 0, 0)
        self.robot_status_value = QLabel("--")
        self.robot_status_value.setStyleSheet("color: #feca57; font-weight: bold;")
        robot_status_layout.addWidget(self.robot_status_value, 0, 1)

        robot_status_layout.addWidget(QLabel("激活状态:"), 0, 2)
        self.robot_activate_value = QLabel("--")
        self.robot_activate_value.setStyleSheet("color: #feca57; font-weight: bold;")
        robot_status_layout.addWidget(self.robot_activate_value, 0, 3)

        robot_status_layout.addWidget(QLabel("运动状态:"), 1, 0)
        self.robot_motion_value = QLabel("--")
        self.robot_motion_value.setStyleSheet("color: #feca57; font-weight: bold;")
        robot_status_layout.addWidget(self.robot_motion_value, 1, 1)

        robot_status_layout.addWidget(QLabel("模式:"), 1, 2)
        self.robot_mode_value = QLabel("--")
        self.robot_mode_value.setStyleSheet("color: #feca57; font-weight: bold;")
        robot_status_layout.addWidget(self.robot_mode_value, 1, 3)

        # 刷新状态按钮
        self.refresh_status_btn = QPushButton("刷新状态")
        self.refresh_status_btn.setEnabled(False)
        self.refresh_status_btn.clicked.connect(self.on_refresh_status)
        robot_status_layout.addWidget(self.refresh_status_btn, 2, 0, 1, 2)

        # 错误信息
        robot_status_layout.addWidget(QLabel("错误信息:"), 2, 2)
        self.robot_error_value = QLabel("无")
        self.robot_error_value.setStyleSheet("color: #2ecc71;")
        robot_status_layout.addWidget(self.robot_error_value, 2, 3)

        left_layout.addWidget(robot_status_group)
        main_layout.addWidget(left_panel, stretch=3)

        # ==========================================
        # 右侧：控制面板
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # --- 连接和状态区域 ---
        conn_group = QGroupBox("连接控制")
        conn_layout = QGridLayout(conn_group)

        conn_layout.addWidget(QLabel("IP:"), 0, 0)
        self.ip_input = QLineEdit(ROBOT_IP)
        self.ip_input.setMaximumWidth(130)
        conn_layout.addWidget(self.ip_input, 0, 1)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.on_connect)
        conn_layout.addWidget(self.connect_btn, 0, 2, 1, 2)  # 跨2列

        self.login_btn = QPushButton("登录")
        self.login_btn.setEnabled(False)
        self.login_btn.clicked.connect(self.on_login)
        conn_layout.addWidget(self.login_btn, 1, 0)

        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.setEnabled(False)
        self.logout_btn.clicked.connect(self.on_logout)
        conn_layout.addWidget(self.logout_btn, 1, 1)

        self.enable_btn = QPushButton("使能")
        self.enable_btn.setEnabled(False)
        self.enable_btn.clicked.connect(self.on_enable)
        conn_layout.addWidget(self.enable_btn, 1, 2)

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

        conn_layout.addWidget(status_frame, 0, 4, 2, 1)
        right_layout.addWidget(conn_group)

        # --- 跟随模式控制 ---
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

        right_layout.addWidget(follower_group)

        # --- 关节位置显示 ---
        joint_group = QGroupBox("关节位置 (度)")
        joint_layout = QGridLayout(joint_group)

        self.joint_labels = []
        self.joint_values = []

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

            self.joint_labels.append(label)
            self.joint_values.append(value)

        right_layout.addWidget(joint_group)

        # --- 关节控制 (Jog) ---
        jog_group = QGroupBox("关节控制 (Jog)")
        jog_layout = QGridLayout(jog_group)

        # 步进选择
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("步进:"))

        self.step_small_btn = QPushButton(f"{JOG_STEP_SMALL}°")
        self.step_small_btn.setCheckable(True)
        self.step_small_btn.clicked.connect(lambda: self.set_jog_step(JOG_STEP_SMALL))
        step_layout.addWidget(self.step_small_btn)

        self.step_medium_btn = QPushButton(f"{JOG_STEP_MEDIUM}°")
        self.step_medium_btn.setCheckable(True)
        self.step_medium_btn.setChecked(True)
        self.step_medium_btn.clicked.connect(lambda: self.set_jog_step(JOG_STEP_MEDIUM))
        step_layout.addWidget(self.step_medium_btn)

        self.step_large_btn = QPushButton(f"{JOG_STEP_LARGE}°")
        self.step_large_btn.setCheckable(True)
        self.step_large_btn.clicked.connect(lambda: self.set_jog_step(JOG_STEP_LARGE))
        step_layout.addWidget(self.step_large_btn)

        jog_layout.addLayout(step_layout, 0, 0, 1, 4)

        # 关节控制按钮 (两列布局)
        self.jog_minus_btns = []
        self.jog_plus_btns = []

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

        right_layout.addWidget(jog_group)

        # --- 预设位置 ---
        preset_group = QGroupBox("预设位置")
        preset_layout = QGridLayout(preset_group)

        self.preset_btns = []
        row, col = 0, 0
        for name, joints in PRESET_POSITIONS.items():
            btn = QPushButton(name)
            btn.setMinimumHeight(40)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, j=joints: self.move_to_preset(j))
            preset_layout.addWidget(btn, row, col)
            self.preset_btns.append(btn)
            col += 1
            if col > 1:
                col = 0
                row += 1

        right_layout.addWidget(preset_group)

        # --- 遥操作控制 (UDP 发送位姿) ---
        teleop_group = QGroupBox("遥操作控制 (步进测试)")
        teleop_layout = QGridLayout(teleop_group)

        # UDP 连接按钮
        self.udp_connect_btn = QPushButton("连接 UDP (9998)")
        self.udp_connect_btn.setEnabled(False)
        self.udp_connect_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.udp_connect_btn.clicked.connect(self.on_udp_connect)
        teleop_layout.addWidget(self.udp_connect_btn, 0, 0, 1, 2)

        # UDP 状态
        self.udp_status = QLabel("●")
        self.udp_status.setStyleSheet("color: gray; font-size: 16px;")
        teleop_layout.addWidget(QLabel("UDP:"), 0, 2)
        teleop_layout.addWidget(self.udp_status, 0, 3)

        # 读取当前位置按钮 (保留，用于观察当前坐标)
        self.read_current_btn = QPushButton("获取当前基座坐标")
        self.read_current_btn.setEnabled(False)
        self.read_current_btn.clicked.connect(self.on_read_current_pose)
        teleop_layout.addWidget(self.read_current_btn, 1, 0, 1, 4)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        teleop_layout.addWidget(line, 2, 0, 1, 4)

        # --- 步进点位测试 (Step-by-Step) ---
        test_btns_layout = QGridLayout()
        
        # 第一行：正向移动
        self.test_p1_btn = QPushButton("X +20")
        self.test_p1_btn.setEnabled(False)
        self.test_p1_btn.clicked.connect(self.on_test_p1)
        test_btns_layout.addWidget(self.test_p1_btn, 0, 0)
        
        self.test_p2_btn = QPushButton("Y +20")
        self.test_p2_btn.setEnabled(False)
        self.test_p2_btn.clicked.connect(self.on_test_p2)
        test_btns_layout.addWidget(self.test_p2_btn, 0, 1)
        
        self.test_p3_btn = QPushButton("Z +20")
        self.test_p3_btn.setEnabled(False)
        self.test_p3_btn.clicked.connect(self.on_test_p3)
        test_btns_layout.addWidget(self.test_p3_btn, 0, 2)
        
        # 第二行：负向移动
        self.test_p1_neg_btn = QPushButton("X -20")
        self.test_p1_neg_btn.setEnabled(False)
        self.test_p1_neg_btn.clicked.connect(self.on_test_p1_neg)
        test_btns_layout.addWidget(self.test_p1_neg_btn, 1, 0)
        
        self.test_p2_neg_btn = QPushButton("Y -20")
        self.test_p2_neg_btn.setEnabled(False)
        self.test_p2_neg_btn.clicked.connect(self.on_test_p2_neg)
        test_btns_layout.addWidget(self.test_p2_neg_btn, 1, 1)
        
        self.test_p3_neg_btn = QPushButton("Z -20")
        self.test_p3_neg_btn.setEnabled(False)
        self.test_p3_neg_btn.clicked.connect(self.on_test_p3_neg)
        test_btns_layout.addWidget(self.test_p3_neg_btn, 1, 2)
        
        teleop_layout.addLayout(test_btns_layout, 3, 0, 1, 4)
        
        # 重置回零按钮
        self.reset_zero_btn = QPushButton("累积量归零 (Reset to Start)")
        self.reset_zero_btn.setEnabled(False)
        self.reset_zero_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; min-height: 35px;")
        self.reset_zero_btn.clicked.connect(self.on_reset_zero)
        teleop_layout.addWidget(self.reset_zero_btn, 4, 0, 1, 4)

        # 说明
        note = QLabel("提示: 按钮发送相对于启动点的累积增量")
        note.setStyleSheet("color: #888; font-size: 9px;")
        teleop_layout.addWidget(note, 5, 0, 1, 4)

        right_layout.addWidget(teleop_group)

        # --- 速度控制 ---
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
        self.vel_slider.valueChanged.connect(self.on_velocity_changed)
        vel_layout.addWidget(self.vel_slider)

        vel_btn_layout = QHBoxLayout()
        for pct in [10, 25, 50, 75, 100]:
            btn = QPushButton(f"{pct}%")
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, v=pct: self.set_velocity(v))
            vel_btn_layout.addWidget(btn)
        vel_layout.addLayout(vel_btn_layout)

        right_layout.addWidget(vel_group)
        right_layout.addStretch()

        main_layout.addWidget(right_panel, stretch=2)

        # ==========================================
        # 状态栏
        # ==========================================
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("请先连接机器人")

        # 样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                min-height: 25px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)

    def connect_signals(self):
        """连接信号和槽"""
        self.signals.joints_updated.connect(self.update_joint_display)
        self.signals.status_updated.connect(self.update_status)
        self.signals.connection_changed.connect(self.on_connection_changed)
        self.signals.command_finished.connect(self.on_command_finished)
        self.signals.error_occurred.connect(self.on_error)
        self.signals.robot_status_updated.connect(self.update_robot_status_display)

    def update_3d_view(self):
        """更新 3D 视图"""
        with self.controller.data_lock:
            joints = self.controller.current_joints.copy()
            actual_tcp = self.controller.actual_tcp.copy() if self.controller.actual_tcp else None

        self.robot_view.update_robot(joints)

        # 更新 TCP 位置显示
        if actual_tcp and len(actual_tcp) >= 6:
            # 使用真实机器人返回的 TCP 位置
            self.tcp_labels['X'].setText(f"{actual_tcp[0]:.2f}")
            self.tcp_labels['Y'].setText(f"{actual_tcp[1]:.2f}")
            self.tcp_labels['Z'].setText(f"{actual_tcp[2]:.2f}")
            self.tcp_labels['Rx'].setText(f"{actual_tcp[3]:.2f}")
            self.tcp_labels['Ry'].setText(f"{actual_tcp[4]:.2f}")
            self.tcp_labels['Rz'].setText(f"{actual_tcp[5]:.2f}")
        else:
            # 使用模型计算的 TCP 位置 (可能与实际有偏差)
            try:
                joints_rad = np.deg2rad(joints)
                T = self.robot_view.robot.fkine(joints_rad)
                pos = T.t * 1000  # mm
                # 使用 ZYX 欧拉角顺序 (与大多数示教器一致)
                rpy = T.rpy(unit='deg', order='zyx')

                self.tcp_labels['X'].setText(f"{pos[0]:.2f}*")
                self.tcp_labels['Y'].setText(f"{pos[1]:.2f}*")
                self.tcp_labels['Z'].setText(f"{pos[2]:.2f}*")
                self.tcp_labels['Rx'].setText(f"{rpy[0]:.2f}*")
                self.tcp_labels['Ry'].setText(f"{rpy[1]:.2f}*")
                self.tcp_labels['Rz'].setText(f"{rpy[2]:.2f}*")
            except:
                pass

    # ==========================================
    # 槽函数
    # ==========================================

    def on_connect(self):
        """连接按钮点击"""
        ip = self.ip_input.text().strip()
        if not ip:
            QMessageBox.warning(self, "警告", "请输入IP地址")
            return

        self.statusBar.showMessage("正在连接...")
        self.connect_btn.setEnabled(False)

        def connect_task():
            success = self.controller.connect(ip, PORT_CONTROL)
            if success:
                self.controller.start_monitoring()

        threading.Thread(target=connect_task, daemon=True).start()

    def on_login(self):
        """登录按钮点击"""
        self.statusBar.showMessage("正在登录...")

        def login_task():
            self.controller.login()

        threading.Thread(target=login_task, daemon=True).start()

    def on_logout(self):
        """退出登录按钮点击"""
        self.statusBar.showMessage("正在退出登录...")

        def logout_task():
            self.controller.logout()

        threading.Thread(target=logout_task, daemon=True).start()

    def on_enable(self):
        """使能按钮点击"""
        self.statusBar.showMessage("正在使能...")

        def enable_task():
            self.controller.enable()

        threading.Thread(target=enable_task, daemon=True).start()

    # --- 跟随模式独立命令 ---

    def on_cmd_start_follower(self):
        """start_follower 命令"""
        self.statusBar.showMessage("执行 start_follower...")

        def task():
            self.controller.cmd_start_follower()

        threading.Thread(target=task, daemon=True).start()

    def on_cmd_set_jog_coordinate(self):
        """set_jog_coordinate --tool 命令 (工具坐标系)"""
        self.statusBar.showMessage("执行 set_jog_coordinate --tool...")

        def task():
            if self.controller.cmd_set_jog_coordinate_tool():
                self.coord_value.setText("Tool")

        threading.Thread(target=task, daemon=True).start()

    def on_cmd_follower_cart(self):
        """follower_cart 命令"""
        self.statusBar.showMessage("执行 follower_cart...")

        def task():
            if self.controller.cmd_follower_cart():
                self.signals.status_updated.emit("follower_cart 已启动")

        threading.Thread(target=task, daemon=True).start()

    def on_cmd_stop_follower(self):
        """stop_follower 命令"""
        self.statusBar.showMessage("执行 stop_follower...")

        def task():
            if self.controller.cmd_stop_follower():
                self.signals.status_updated.emit("跟随模式已停止")

        threading.Thread(target=task, daemon=True).start()

    def on_one_click_follower(self):
        """一键启动跟随模式"""
        self.statusBar.showMessage("正在启动跟随模式...")
        self.one_click_follower_btn.setEnabled(False)

        def task():
            ip = self.ip_input.text().strip()
            success = self.controller.start_follower_mode(ip)
            if success:
                self.signals.command_finished.emit(True, "跟随模式已启动")
            else:
                self.signals.command_finished.emit(False, "跟随模式启动失败")

        threading.Thread(target=task, daemon=True).start()

    def on_connection_changed(self, connected, port_type):
        """连接状态变化"""
        if port_type == "websocket":
            color = "lime" if connected else "red"
            self.ws_status.setStyleSheet(f"color: {color}; font-size: 16px;")
            if connected:
                self.login_btn.setEnabled(True)
                self.connect_btn.setEnabled(True)

        self.statusBar.showMessage("连接成功" if connected else "连接断开")

    def on_command_finished(self, success, message):
        """命令执行完成"""
        self.statusBar.showMessage(message)
        # 更新跟随模式按钮状态
        if "跟随模式" in message:
            self.one_click_follower_btn.setEnabled(True)
            if "已启动" in message and success:
                self.update_status("跟随模式已启动")
            elif "已停止" in message:
                self.update_status("跟随模式已停止")

        # 连续运动完成后恢复按钮状态
        if "连续运动" in message:
            self.stop_continuous_btn.setEnabled(False)
            # 如果跟随模式和UDP都连接，重新启用连续运动按钮
            if self.controller.is_follower_mode and self.controller.udp_connected:
                self.continuous_motion_btn.setEnabled(True)

    def on_error(self, error_msg):
        """错误处理"""
        self.statusBar.showMessage(f"错误: {error_msg}")
        self.connect_btn.setEnabled(True)

    def update_status(self, message):
        """更新状态消息"""
        self.statusBar.showMessage(message)

        if "登录成功" in message:
            self.login_status.setStyleSheet("color: lime; font-size: 16px;")
            self.enable_btn.setEnabled(True)
            self.logout_btn.setEnabled(True)
            self.refresh_status_btn.setEnabled(True)

        if "已退出登录" in message:
            self.login_status.setStyleSheet("color: gray; font-size: 16px;")
            self.enable_status.setStyleSheet("color: gray; font-size: 16px;")
            self.follower_status.setStyleSheet("color: gray; font-size: 16px;")
            self.enable_btn.setEnabled(False)
            self.logout_btn.setEnabled(False)
            self.refresh_status_btn.setEnabled(False)
            self.enable_jog_controls(False)
            # 禁用跟随模式相关按钮
            self.start_follower_btn.setEnabled(False)
            self.set_jog_coord_btn.setEnabled(False)
            self.follower_cart_btn.setEnabled(False)
            self.stop_follower_btn.setEnabled(False)
            self.one_click_follower_btn.setEnabled(False)

        if "使能成功" in message:
            self.enable_status.setStyleSheet("color: lime; font-size: 16px;")
            self.enable_jog_controls(True)
            # 使能后启用跟随模式相关按钮
            self.start_follower_btn.setEnabled(True)
            self.set_jog_coord_btn.setEnabled(True)
            self.follower_cart_btn.setEnabled(True)
            self.stop_follower_btn.setEnabled(True)
            self.one_click_follower_btn.setEnabled(True)
            # 启用 UDP 连接按钮
            self.udp_connect_btn.setEnabled(True)
            self.read_current_btn.setEnabled(True)

        if "跟随模式已启动" in message or "follower_cart 已启动" in message:
            self.follower_status.setStyleSheet("color: lime; font-size: 16px;")
            self.one_click_follower_btn.setEnabled(True)
            self.coord_value.setText("Tool")  # 使用工具坐标系 (Eye-in-Hand)
            # 跟随模式启动后，如果 UDP 已连接则启用发送按钮和测试按钮
            if self.controller.udp_connected:
                self.send_pose_btn.setEnabled(True)
                self.test_p1_btn.setEnabled(True)
                self.test_p2_btn.setEnabled(True)
                self.test_p3_btn.setEnabled(True)
                self.test_p1_neg_btn.setEnabled(True)
                self.test_p2_neg_btn.setEnabled(True)
                self.test_p3_neg_btn.setEnabled(True)
                self.reset_zero_btn.setEnabled(True)

        if "跟随模式已停止" in message:
            self.follower_status.setStyleSheet("color: gray; font-size: 16px;")
            self.one_click_follower_btn.setEnabled(True)
            self.coord_value.setText("Joint")
            # 跟随模式停止，禁用发送按钮和测试按钮
            self.send_pose_btn.setEnabled(False)
            self.test_p1_btn.setEnabled(False)
            self.test_p2_btn.setEnabled(False)
            self.test_p3_btn.setEnabled(False)
            self.test_p1_neg_btn.setEnabled(False)
            self.test_p2_neg_btn.setEnabled(False)
            self.test_p3_neg_btn.setEnabled(False)
            self.reset_zero_btn.setEnabled(False)

        if "UDP 遥操作连接成功" in message:
            self.udp_status.setStyleSheet("color: lime; font-size: 16px;")
            self.udp_connect_btn.setText("断开 UDP")
            # 如果跟随模式已启动，启用发送按钮和测试按钮
            if self.controller.is_follower_mode:
                self.send_pose_btn.setEnabled(True)
                self.test_p1_btn.setEnabled(True)
                self.test_p2_btn.setEnabled(True)
                self.test_p3_btn.setEnabled(True)
                self.test_p1_neg_btn.setEnabled(True)
                self.test_p2_neg_btn.setEnabled(True)
                self.test_p3_neg_btn.setEnabled(True)
                self.reset_zero_btn.setEnabled(True)

        if "UDP 连接已断开" in message:
            self.udp_status.setStyleSheet("color: gray; font-size: 16px;")
            self.udp_connect_btn.setText("连接 UDP (9998)")
            self.send_pose_btn.setEnabled(False)
            self.test_p1_btn.setEnabled(False)
            self.test_p2_btn.setEnabled(False)
            self.test_p3_btn.setEnabled(False)
            self.test_p1_neg_btn.setEnabled(False)
            self.test_p2_neg_btn.setEnabled(False)
            self.test_p3_neg_btn.setEnabled(False)
            self.reset_zero_btn.setEnabled(False)

    def update_joint_display(self, joints):
        """更新关节位置显示"""
        for i, val in enumerate(joints[:6]):
            self.joint_values[i].setText(f"{val:.3f}")

    def enable_jog_controls(self, enabled):
        """启用/禁用Jog控制"""
        for btn in self.jog_minus_btns + self.jog_plus_btns + self.preset_btns:
            btn.setEnabled(enabled)
        self.vel_slider.setEnabled(enabled)

    def set_jog_step(self, step):
        """设置Jog步进"""
        self.jog_step = step
        self.step_small_btn.setChecked(step == JOG_STEP_SMALL)
        self.step_medium_btn.setChecked(step == JOG_STEP_MEDIUM)
        self.step_large_btn.setChecked(step == JOG_STEP_LARGE)

    def jog_joint(self, joint_index, direction):
        """Jog单个关节"""
        self.controller.jog_joint(joint_index, direction, self.jog_step)

    def move_to_preset(self, joints):
        """移动到预设位置"""
        self.controller.move_joint(joints)

    def on_velocity_changed(self, value):
        """速度滑块值变化"""
        self.vel_value_label.setText(f"{value}%")

    def set_velocity(self, percent):
        """设置速度"""
        self.vel_slider.setValue(percent)
        self.controller.set_velocity(percent)

    # --- 遥操作相关槽函数 ---

    def on_udp_connect(self):
        """UDP 连接/断开按钮点击"""
        if self.controller.udp_connected:
            # 断开连接
            self.controller.disconnect_udp()
        else:
            # 连接
            self.statusBar.showMessage("正在连接 UDP...")

            def connect_task():
                self.controller.connect_udp(9998)

            threading.Thread(target=connect_task, daemon=True).start()

    def on_send_pose(self):
        """发送目标位姿"""
        try:
            x = float(self.target_x.text())
            y = float(self.target_y.text())
            z = float(self.target_z.text())
            rz = float(self.target_rz.text())
            ry = float(self.target_ry.text())
            rx = float(self.target_rx.text())

            self.statusBar.showMessage("正在发送位姿...")

            def send_task():
                self.controller.send_target_pose(
                    x=x, y=y, z=z,
                    rz=rz, ry=ry, rx=rx,
                    unit_mm=True, unit_deg=True
                )

            threading.Thread(target=send_task, daemon=True).start()

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数值")

    def on_read_current_pose(self):
        """读取当前位置填入输入框"""
        tcp = self.controller.get_current_tcp_mm_deg()
        if tcp is not None:
            x, y, z, rx, ry, rz = tcp
            self.target_x.setText(f"{x:.1f}")
            self.target_y.setText(f"{y:.1f}")
            self.target_z.setText(f"{z:.1f}")
            # 注意: 机器人返回的可能是 XYZ 顺序，需要调整为 ZYX 显示
            self.target_rx.setText(f"{rx:.1f}")
            self.target_ry.setText(f"{ry:.1f}")
            self.target_rz.setText(f"{rz:.1f}")
            self.statusBar.showMessage("已读取当前位置")
        else:
            self.statusBar.showMessage("无法获取当前位置")

    def on_test_p1(self):
        """测试点 1: X +20mm"""
        self.statusBar.showMessage("发送测试指令: X +20mm")
        def task():
            # 发送 X+20mm 增量 (累积)
            self.controller.send_raw_increment(20.0, 0, 0)
        threading.Thread(target=task, daemon=True).start()

    def on_test_p2(self):
        """测试点 2: Y +20mm"""
        self.statusBar.showMessage("发送测试指令: Y +20mm")
        def task():
            # 发送 Y+20mm 增量 (累积)
            self.controller.send_raw_increment(0, 20.0, 0)
        threading.Thread(target=task, daemon=True).start()

    def on_test_p3(self):
        """测试点 3: Z +20mm"""
        self.statusBar.showMessage("发送测试指令: Z +20mm")
        def task():
            # 发送 Z+20mm 增量 (累积)
            self.controller.send_raw_increment(0, 0, 20.0)
        threading.Thread(target=task, daemon=True).start()

    def on_test_p1_neg(self):
        """测试点 1 负向: X -20mm"""
        self.statusBar.showMessage("发送测试指令: X -20mm")
        def task():
            self.controller.send_raw_increment(-20.0, 0, 0)
        threading.Thread(target=task, daemon=True).start()

    def on_test_p2_neg(self):
        """测试点 2 负向: Y -20mm"""
        self.statusBar.showMessage("发送测试指令: Y -20mm")
        def task():
            self.controller.send_raw_increment(0, -20.0, 0)
        threading.Thread(target=task, daemon=True).start()

    def on_test_p3_neg(self):
        """测试点 3 负向: Z -20mm"""
        self.statusBar.showMessage("发送测试指令: Z -20mm")
        def task():
            self.controller.send_raw_increment(0, 0, -20.0)
        threading.Thread(target=task, daemon=True).start()

    def on_reset_zero(self):
        """重置累积量为零"""
        self.statusBar.showMessage("重置累积偏移量...")
        def task():
            # 1. 本地重置
            self.controller.follower_offset = np.zeros(6)
            # 2. 发送零位包给机器人，确保机器人也回到跟随起点
            # 注意：send_pose_euler 发送的是绝对偏移量 (现在是0)
            success = self.controller.udp_client.send_pose_euler(0, 0, 0, 0, 0, 0, "321")
            if success:
                self.signals.status_updated.emit("累积量已重置为 0")
        threading.Thread(target=task, daemon=True).start()

    def on_refresh_status(self):
        """刷新机器人状态"""
        self.statusBar.showMessage("正在查询状态...")

        def refresh_task():
            self.controller.get_robot_status()

        threading.Thread(target=refresh_task, daemon=True).start()

    def update_robot_status_display(self, status_info):
        """更新机器人状态显示"""
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
            # 同步更新右上角使能状态指示灯
            self.enable_status.setStyleSheet("color: lime; font-size: 16px;")
        else:
            self.robot_activate_value.setStyleSheet("color: #feca57; font-weight: bold;")
            # 同步更新右上角使能状态指示灯
            if activate == 'Disabled':
                self.enable_status.setStyleSheet("color: red; font-size: 16px;")

        # 运动状态
        motion = status_info.get('motion', '--')
        self.robot_motion_value.setText(motion)
        if motion == 'Running':
            self.robot_motion_value.setStyleSheet("color: #2ecc71; font-weight: bold;")
            # 同步更新右上角跟随状态指示灯 (如果在跟随模式)
            if self.controller.is_follower_mode:
                self.follower_status.setStyleSheet("color: lime; font-size: 16px;")
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

        self.statusBar.showMessage("状态已更新")

    def on_move_to_target(self):
        """移动到 3D 视图中的目标点 (红点)"""
        if not self.CONTINUOUS_TEST_POINTS:
            QMessageBox.warning(self, "错误", "没有定义目标点")
            return
            
        target = self.CONTINUOUS_TEST_POINTS[0]
        # 格式: (x, y, z, rx, ry, rz) 单位: mm, deg
        x, y, z, rx, ry, rz = target
        
        self.statusBar.showMessage(f"正在解算 IK: ({x}, {y}, {z})...")
        
        try:
            # 1. 构建目标位姿矩阵 (SE3)
            # 假设 rx, ry, rz 对应 ZYX 欧拉角顺序 (先绕Z，再绕Y，最后绕X)
            # 位置转为米
            T = SE3(x/1000.0, y/1000.0, z/1000.0) * \
                SE3.Rz(rz, unit='deg') * \
                SE3.Ry(ry, unit='deg') * \
                SE3.Rx(rx, unit='deg')
            
            # 2. 进行逆运动学解算 (IK)
            # 使用 Levenberg-Marquardt 数值解法
            # q0 可以设为当前关节角作为初值，加快收敛
            import time
            current_q_rad = np.deg2rad(self.controller.current_joints)
            
            t_start = time.perf_counter()
            sol = self.robot_view.robot.ikine_LM(T, q0=current_q_rad)
            t_end = time.perf_counter()
            ik_time = (t_end - t_start) * 1000 # ms
            
            print(f"[IK] 单点解算耗时: {ik_time:.2f} ms")
            
            if not sol.success:
                QMessageBox.warning(self, "IK 失败", f"无法找到该位姿的逆解!\n原因: {sol.reason}\n耗时: {ik_time:.2f} ms")
                return
                
            # 3. 提取关节角并转为度
            target_joints_deg = np.rad2deg(sol.q).tolist()
            
            # 4. 确认是否移动
            msg = f"已解算出关节角 (耗时 {ik_time:.2f} ms):\n{['%.2f'%j for j in target_joints_deg]}\n\n是否立即移动机器人?"
            reply = QMessageBox.question(self, "确认移动", msg, QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.statusBar.showMessage("正在移动到目标点...")
                # 设定速度 (可以稍微慢点)
                vels = [20, 20, 20] # 20% 速度
                
                def move_task():
                    # 调用 controller 的 move_joint (注意: 需要传入 vels 参数)
                    self.controller.move_joint(target_joints_deg, vels)
                
                threading.Thread(target=move_task, daemon=True).start()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算或执行出错: {e}")

    def on_run_trajectory(self):
        """生成并执行圆形轨迹"""
        if not self.CONTINUOUS_TEST_POINTS:
            QMessageBox.warning(self, "错误", "没有定义参考点")
            return
            
        center = self.CONTINUOUS_TEST_POINTS[0] # (x, y, z, rx, ry, rz)
        cx, cy, cz, rx, ry, rz = center
        
        # 参数: 半径 50mm, 12个点
        radius = 50.0 
        num_points = 12
        
        # 生成圆周点 (在 XY 平面)
        trajectory_points = []
        # 生成 0 到 360 度的点
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        # 为了闭合轨迹，添加第一个点到末尾
        angles = np.append(angles, 0)
        
        for theta in angles:
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = cz
            trajectory_points.append((x, y, z))
            
        # 1. 在 3D 视图绘制轨迹线 (青色)
        self.robot_view.set_trajectory(trajectory_points)
        
        # 2. 确认执行
        reply = QMessageBox.question(self, "确认执行", 
            f"已生成 {len(trajectory_points)} 个点的圆形轨迹 (半径 {radius}mm).\n"
            f"将在 3D 视图中显示青色轨迹线。\n\n是否开始执行?",
            QMessageBox.Yes | QMessageBox.No)
            
        if reply != QMessageBox.Yes:
            return
            
        # 3. 启动执行线程
        self.statusBar.showMessage("正在执行轨迹运动...")
        
        def run_task():
            import time
            # 速度设置 (慢速，确保安全)
            vels = [15, 15, 15]
            
            # 使用当前的关节角作为 IK 初值，保证连续性
            last_q_rad = np.deg2rad(self.controller.current_joints)
            
            total = len(trajectory_points)
            for i, (tx, ty, tz) in enumerate(trajectory_points):
                self.signals.status_updated.emit(f"正在前往轨迹点 {i+1}/{total}...")
                
                # 构建位姿 (保持姿态不变，只改变位置)
                T = SE3(tx/1000.0, ty/1000.0, tz/1000.0) * \
                    SE3.Rz(rz, unit='deg') * \
                    SE3.Ry(ry, unit='deg') * \
                    SE3.Rx(rx, unit='deg')
                
                # IK 解算 (使用上一帧解作为初值 q0)
                t_start = time.perf_counter()
                sol = self.robot_view.robot.ikine_LM(T, q0=last_q_rad)
                t_end = time.perf_counter()
                ik_time = (t_end - t_start) * 1000 # ms
                
                if sol.success:
                    print(f"[IK] 点 {i+1} 解算耗时: {ik_time:.2f} ms")
                    
                    target_joints = np.rad2deg(sol.q).tolist()
                    last_q_rad = sol.q # 更新初值
                    
                    # 发送指令
                    # move_joint 内部是阻塞调用(等待机器人回复)，所以天然同步
                    self.controller.move_joint(target_joints, vels)
                    
                    # 稍微停顿，展示效果
                    time.sleep(0.5) 
                else:
                    self.signals.error_occurred.emit(f"轨迹点 {i+1} IK解算失败! 耗时: {ik_time:.2f} ms")
                    break
            
            self.signals.status_updated.emit("轨迹执行完成")
            
        threading.Thread(target=run_task, daemon=True).start()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.view_timer.stop()
        self.controller.stop()
        # 关闭 PyVista 渲染器
        try:
            self.robot_view.plotter.close()
        except:
            pass
        event.accept()
