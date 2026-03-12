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
from .ui.log_panel import LogPanel
from .ui.trajectory_generator_panel import TrajectoryGeneratorPanel
from .logic.conveyor_tracking_service import ConveyorTrackingService

# 导入逻辑服务
from . import config
import sys

class StreamRedirector:
    """更安全的日志重定向器 (带防递归保护)"""
    def __init__(self, signal):
        self.signal = signal
        self._lock = threading.Lock()
        self._is_writing = False

    def write(self, text):
        if not text: return 0
        # 防递归检查：如果已经在写入过程中触发了打印，直接跳过
        if self._is_writing:
            return len(text)
            
        with self._lock:
            self._is_writing = True
            try:
                msg = str(text).strip()
                if msg:
                    self.signal.emit(msg)
            except:
                pass
            finally:
                self._is_writing = False
        return len(text)

    def flush(self):
        pass

class TeachPendantWindow(QMainWindow):
    """示教器主窗口 - 工业级全屏界面"""

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.controller = RobotController(self.signals)
        self.fast_ik = FastIKSolver()
        
        self.init_ui()
        self.connect_signals()
        
        # 初始化传送带追踪服务
        self.tracking_service = ConveyorTrackingService(self.controller, self.fast_ik, self.robot_view)
        
        self.tracking_service.status_updated.connect(self.update_status)
        
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

        # 最后：激活日志重定向 (更安全的方法)
        self._setup_log_redirection()

    def _setup_log_redirection(self):
        """安全地设置日志重定向 (仅限标准输出)"""
        try:
            # 只重定向标准输出，保留标准错误(stderr)到终端，防止闪退
            sys.stdout = StreamRedirector(self.signals.log_message)
            print("[System] 日志系统重定向已激活 (Stdout Only)")
        except:
            pass

    def _debug_print_tracking(self):
        # 通过 robot_view (Robot3DWidget) 获取其内部的 renderer
        if hasattr(self.robot_view, 'renderer') and self.robot_view.renderer:
            target, target_id = self.robot_view.renderer.get_tracking_target()
            if target is not None:
                print(f"[阶段一验证] 锁定目标 ID={target_id}: X={target[0]:.3f}, Y={target[1]:.3f}, Z={target[2]:.3f}")

    def _on_conveyor_speed_changed(self, speed):
        """同步更新渲染器和追踪服务的传送带速度"""
        if hasattr(self.robot_view, 'renderer') and self.robot_view.renderer:
            self.robot_view.renderer.belt_speed = speed
        
        if hasattr(self, 'tracking_service'):
            self.tracking_service.set_conveyor_speed(speed)
        
        self.statusBar.showMessage(f"传送带速度已调整为: {speed} m/s")

    def _on_conveyor_tracking_toggled(self, checked):
        """处理来自 VisionPanel 的追踪开关请求"""
        if checked:
            # 只有在机器人使能的情况下才允许真正追踪
            if not self.controller.state.is_enabled:
                QMessageBox.warning(self, "警告", "机器人未使能，请先连接并使能机器人！")
                # 强制将按钮状态改回去 (不触发信号递归)
                self.vision_panel.tracking_btn.blockSignals(True)
                self.vision_panel.tracking_btn.setChecked(False)
                self.vision_panel.on_tracking_toggled(False)
                self.vision_panel.tracking_btn.blockSignals(False)
                return

            self.tracking_service.hover_only_mode = False
            self.tracking_service.use_sim_algorithm = False
            self.tracking_service.use_dynamic_offset = False
            self.tracking_service.start_tracking()
        else:
            self.tracking_service.stop_tracking()

    def _on_conveyor_sim_tracking_toggled(self, checked):
        """处理来自 VisionPanel 的仿真速度PI追踪开关请求"""
        if checked:
            # 只有在机器人使能的情况下才允许真正追踪
            if not self.controller.state.is_enabled:
                QMessageBox.warning(self, "警告", "机器人未使能，请先连接并使能机器人！")
                # 强制将按钮状态改回去 (不触发信号递归)
                self.vision_panel.sim_tracking_btn.blockSignals(True)
                self.vision_panel.sim_tracking_btn.setChecked(False)
                self.vision_panel.on_sim_tracking_toggled(False)
                self.vision_panel.sim_tracking_btn.blockSignals(False)
                return

            self.tracking_service.hover_only_mode = False
            self.tracking_service.use_sim_algorithm = True
            self.tracking_service.use_dynamic_offset = False
            self.tracking_service.start_tracking()
        else:
            self.tracking_service.use_sim_algorithm = False
            self.tracking_service.stop_tracking()

    def _on_conveyor_offset_tracking_toggled(self, checked):
        """处理来自 VisionPanel 的自适应隐性Offset追踪开关请求"""
        if checked:
            # 只有在机器人使能的情况下才允许真正追踪
            if not self.controller.state.is_enabled:
                QMessageBox.warning(self, "警告", "机器人未使能，请先连接并使能机器人！")
                # 强制将按钮状态改回去 (不触发信号递归)
                self.vision_panel.offset_tracking_btn.blockSignals(True)
                self.vision_panel.offset_tracking_btn.setChecked(False)
                self.vision_panel.on_offset_tracking_toggled(False)
                self.vision_panel.offset_tracking_btn.blockSignals(False)
                return

            self.tracking_service.hover_only_mode = False
            self.tracking_service.use_sim_algorithm = False
            self.tracking_service.use_dynamic_offset = True
            self.tracking_service.start_tracking()
        else:
            self.tracking_service.use_dynamic_offset = False
            self.tracking_service.stop_tracking()

    def _on_conveyor_udp_follower_tracking_toggled(self, checked):
        """处理来自 VisionPanel 的 UDP follower_cart 追踪开关请求"""
        if checked:
            if not self.controller.state.is_enabled:
                QMessageBox.warning(self, "警告", "机器人未使能！")
                self.vision_panel.udp_follower_tracking_btn.blockSignals(True)
                self.vision_panel.udp_follower_tracking_btn.setChecked(False)
                self.vision_panel.on_udp_follower_tracking_toggled(False)
                self.vision_panel.udp_follower_tracking_btn.blockSignals(False)
                return
            self.tracking_service.use_udp_follower = True
            self.tracking_service.start_tracking()
        else:
            self.tracking_service.stop_tracking()
            self.tracking_service.use_udp_follower = False

    def _on_get_current_position_requested(self):
        """处理获取当前坐标请求"""
        if not self.controller.state.is_enabled:
            QMessageBox.warning(self, "警告", "机器人未使能，请先连接并使能机器人！")
            return
        
        # 使用 controller 获取当前 TCP（已通过 follower_client 修正旋转顺序）
        tcp = self.controller.get_current_tcp_mm_deg()
        if tcp:
            self.vision_panel.update_current_position_display(
                tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5]
            )
            self.statusBar.showMessage(f"当前坐标: X={tcp[0]:.2f}, Y={tcp[1]:.2f}, Z={tcp[2]:.2f}, Rx={tcp[3]:.2f}, Ry={tcp[4]:.2f}, Rz={tcp[5]:.2f}")
        else:
            QMessageBox.warning(self, "警告", "无法获取当前坐标")

    def _on_single_point_move_requested(self, target_x, target_y, target_z, target_rx, target_ry, target_rz):
        """
        处理单点遥操作移动请求
        从当前位置移动到目标绝对坐标位置
        """
        # 检查机器人状态
        if not self.controller.state.is_enabled:
            QMessageBox.warning(self, "警告", "机器人未使能，请先连接并使能机器人！")
            return
        
        # 获取当前位置作为原点
        origin_tcp = self.controller.get_current_tcp_mm_deg()
        if not origin_tcp:
            QMessageBox.warning(self, "警告", "无法获取当前坐标，请先点击'获取当前坐标'")
            return
        
        # 计算偏移量
        offset_x = target_x - origin_tcp[0]
        offset_y = target_y - origin_tcp[1]
        offset_z = target_z - origin_tcp[2]
        offset_rx = target_rx - origin_tcp[3]
        offset_ry = target_ry - origin_tcp[4]
        offset_rz = target_rz - origin_tcp[5]
        
        # 确认对话框
        msg = f"当前位置: X={origin_tcp[0]:.2f}, Y={origin_tcp[1]:.2f}, Z={origin_tcp[2]:.2f}\n" \
              f"目标位置: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}\n" \
              f"偏移量: X={offset_x:+.2f}, Y={offset_y:+.2f}, Z={offset_z:+.2f}\n\n" \
              f"是否执行移动？"
        
        if QMessageBox.question(self, "确认移动", msg) != QMessageBox.Yes:
            return
        
        # 在后台线程中执行移动
        threading.Thread(
            target=self._execute_single_point_move,
            args=(origin_tcp, target_x, target_y, target_z, target_rx, target_ry, target_rz),
            daemon=True
        ).start()

    def _execute_single_point_move(self, origin_tcp, target_x, target_y, target_z, target_rx, target_ry, target_rz):
        """
        执行单点遥操作移动（在后台线程中运行）
        """
        try:
            # 1. 确保UDP已连接
            if not self.controller.udp_connected:
                self.signals.status_updated.emit("正在连接 UDP...")
                if not self.controller.connect_udp():
                    self.signals.error_occurred.emit("UDP 连接失败")
                    return
                time.sleep(0.5)
            
            # 2. 确保跟随模式已启动
            if not self.controller.is_follower_mode:
                self.signals.status_updated.emit("正在启动跟随模式...")
                ip = self.conn_panel.get_ip()
                if not self.controller.start_follower_mode(ip):
                    self.signals.error_occurred.emit("跟随模式启动失败")
                    return
                time.sleep(1.0)
            
            # 3. 计算偏移量
            offset_x = target_x - origin_tcp[0]
            offset_y = target_y - origin_tcp[1]
            offset_z = target_z - origin_tcp[2]
            offset_rx = target_rx - origin_tcp[3]
            offset_ry = target_ry - origin_tcp[4]
            offset_rz = target_rz - origin_tcp[5]
            
            # 4. 转换到UDP增量坐标系
            # 根据坐标映射测试结果:
            # - send_pose(x,...) 直接控制基座 X (1:1)
            # - send_pose(...,y,...) 控制基座 Y 但需要取反
            # - Z 也需要取反
            send_x = offset_x / 1000.0          # X 直接对应
            send_y = -offset_y / 1000.0         # Y 需要取反  
            send_z = -offset_z / 1000.0         # Z 取反
            # 旋转映射: dry(index 3)=ry, drx(index 4)=rx, drz(index 5)=rz
            dry = np.deg2rad(offset_ry)
            drx = np.deg2rad(offset_rx)
            drz = np.deg2rad(offset_rz)
            
            print(f"\n{'='*60}")
            print("[单点遥操作] 开始移动")
            print(f"[原点] X={origin_tcp[0]:.2f}, Y={origin_tcp[1]:.2f}, Z={origin_tcp[2]:.2f}")
            print(f"[目标] X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")
            print(f"[偏移] X={offset_x:+.2f}, Y={offset_y:+.2f}, Z={offset_z:+.2f}")
            print(f"[UDP参数] x={send_x:.4f}, y={send_y:.4f}, z={send_z:.4f}")
            print(f"{'='*60}")
            
            # 5. 设置控制器偏移量 [send_x, send_y, send_z, dry, drx, drz]
            # 与 send_pose_euler 参数顺序一致
            self.controller.follower_offset[:] = [send_x, send_y, send_z, dry, drx, drz]
            
            # 6. 发送增量指令，持续一段时间让机器人到位
            self.signals.status_updated.emit(f"正在移动至目标位置...")
            move_duration = 5.0  # 移动持续时间(秒)
            start_time = time.time()
            count = 0
            
            while time.time() - start_time < move_duration:
                t_loop_start = time.perf_counter()
                
                if not self.controller.state.is_enabled or not self.controller.is_follower_mode:
                    self.signals.error_occurred.emit("机器人状态异常，移动中断")
                    return
                
                # 发送UDP增量
                # 根据坐标映射测试结果: send_pose(x, y, z, ...)
                # x=基座X偏移, y=基座Y偏移(取反), z=基座Z偏移(取反)
                self.controller.udp_client.send_pose_euler(send_x, send_y, send_z, dry, drx, drz)
                count += 1
                
                # 控制发送频率 50Hz
                time.sleep(max(0, 0.02 - (time.perf_counter() - t_loop_start)))
            
            # 7. 获取最终位置
            final_tcp = self.controller.get_current_tcp_mm_deg()
            if final_tcp:
                error = np.linalg.norm([
                    final_tcp[0] - target_x,
                    final_tcp[1] - target_y,
                    final_tcp[2] - target_z
                ])
                print(f"[完成] 最终位置: X={final_tcp[0]:.2f}, Y={final_tcp[1]:.2f}, Z={final_tcp[2]:.2f}")
                print(f"[完成] 位置误差: {error:.2f}mm")
                self.signals.status_updated.emit(f"单点移动完成，误差: {error:.2f}mm")
            else:
                self.signals.status_updated.emit("单点移动完成")
                
        except Exception as e:
            self.signals.error_occurred.emit(f"单点移动异常: {e}")
            import traceback
            traceback.print_exc()

    def _on_conveyor_hover_only_toggled(self, checked):
        """处理来自 VisionPanel 的仅悬停追踪开关请求"""
        if checked:
            # 只有在机器人使能的情况下才允许真正追踪
            if not self.controller.state.is_enabled:
                QMessageBox.warning(self, "警告", "机器人未使能，请先连接并使能机器人！")
                # 强制将按钮状态改回去 (不触发信号递归)
                self.vision_panel.hover_only_btn.blockSignals(True)
                self.vision_panel.hover_only_btn.setChecked(False)
                self.vision_panel.on_hover_only_toggled(False)
                self.vision_panel.hover_only_btn.blockSignals(False)
                return

            self.tracking_service.hover_only_mode = True
            self.tracking_service.start_tracking()
        else:
            self.tracking_service.stop_tracking()

    def init_ui(self):
        """初始化 UI 布局"""
        self.setWindowTitle("仿真孪生系统 - KAANH")
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

        view_group = QGroupBox("佳安 3D 仿真场景")
        view_layout = QVBoxLayout(view_group)
        self.robot_view = Robot3DWidget()
        view_layout.addWidget(self.robot_view)

        view_btn_layout = QHBoxLayout()
        reset_view_btn = QPushButton("重置视角")
        reset_view_btn.clicked.connect(self.robot_view.reset_view)
        view_btn_layout.addWidget(reset_view_btn)
        
        # FOV Toggle
        from PyQt5.QtWidgets import QCheckBox
        self.fov_toggle = QCheckBox("显示摄像头视野")
        self.fov_toggle.setChecked(True)
        self.fov_toggle.stateChanged.connect(lambda state: self.robot_view.set_fov_visible(state == Qt.Checked))
        view_btn_layout.addWidget(self.fov_toggle)
        
        view_btn_layout.addStretch()
        view_layout.addLayout(view_btn_layout)
        left_layout.addWidget(view_group, stretch=3) # 增加 3D 视图权重

        # ==========================================
        # 左侧下方：状态面板 + 日志面板 (并排显示)
        # ==========================================
        bottom_info_layout = QHBoxLayout()
        bottom_info_layout.setSpacing(10)

        # 状态面板 (缩小版)
        self.status_panel = RobotStatusPanel(self.controller, self.signals)
        bottom_info_layout.addWidget(self.status_panel, stretch=1)

        # 日志面板 (新增)
        self.log_panel = LogPanel(self.signals)
        bottom_info_layout.addWidget(self.log_panel, stretch=1)

        left_layout.addLayout(bottom_info_layout, stretch=1)

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
        self.vision_panel.udp_execution_requested.connect(self.on_execute_vision_trajectory_udp)
        self.vision_panel.actual_export_requested.connect(self.on_save_actual_trajectory)
        self.vision_panel.conveyor_tracking_toggled.connect(self._on_conveyor_tracking_toggled)
        self.vision_panel.conveyor_sim_tracking_toggled.connect(self._on_conveyor_sim_tracking_toggled)
        self.vision_panel.conveyor_offset_tracking_toggled.connect(self._on_conveyor_offset_tracking_toggled)
        self.vision_panel.conveyor_hover_only_toggled.connect(self._on_conveyor_hover_only_toggled)
        self.vision_panel.conveyor_speed_changed.connect(self._on_conveyor_speed_changed)
        self.vision_panel.conveyor_udp_follower_tracking_toggled.connect(self._on_conveyor_udp_follower_tracking_toggled)
        self.vision_panel.single_point_move_requested.connect(self._on_single_point_move_requested)
        self.vision_panel.get_current_position_requested.connect(self._on_get_current_position_requested)
        vision_layout.addWidget(self.vision_panel)
        vision_layout.addStretch()
        self.tabs.addTab(vision_tab, "视觉引导")

        # Tab 4: 轨迹生成器
        traj_gen_tab = QWidget()
        traj_gen_layout = QVBoxLayout(traj_gen_tab)
        self.trajectory_gen_panel = TrajectoryGeneratorPanel(self.robot_view, self.controller)
        self.trajectory_gen_panel.trajectory_generated.connect(self.on_manual_trajectory_ready)
        traj_gen_layout.addWidget(self.trajectory_gen_panel)
        traj_gen_layout.addStretch()
        self.tabs.addTab(traj_gen_tab, "轨迹生成")

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
                actual_fixed = np.array([actual_tcp[0], actual_tcp[1], actual_tcp[2] - 50.8])
                lin_err = np.linalg.norm(np.array(p[:3]) - actual_fixed)
                self.signals.tracking_error_updated.emit(lin_err, 0.0)
        else:
            # 回退到 DH 计算预览
            try:
                T = self.robot_view.robot.fkine(np.deg2rad(joints))
                self.status_panel.update_tcp_display(model_pos=T.t*1000, model_rpy=T.rpy(unit='deg', order='zyx'))
            except: pass
        
        # 更新视野内白点位置显示（如果启用了白点检测）
        self._update_white_points_detection()

    def _update_white_points_detection(self):
        """检测视野内的白点并更新UI显示"""
        # 检查是否启用了白点检测
        if not hasattr(self, 'vision_panel') or not self.vision_panel.white_points_toggle_btn.isChecked():
            return
        
        try:
            # 获取渲染器中的白点信息
            renderer = self.robot_view.renderer
            if not hasattr(renderer, 'get_visible_white_points'):
                return
            
            # 获取视野内的白点列表
            white_points = renderer.get_visible_white_points()
            
            # 更新UI显示
            self.vision_panel.update_white_points_display(white_points)
            
        except Exception as e:
            # 静默处理错误，避免影响主循环
            pass

    def on_vision_trajectory_ready(self, points):
        self.current_vision_trajectory = points
        display_points = [(p[0], p[1], p[2]) for p in points]
        self.robot_view.set_trajectory(display_points)
        self.statusBar.showMessage(f"轨迹加载成功，共 {len(points)} 个点位")

    def on_manual_trajectory_ready(self, points, time_step=None):
        """手动生成的轨迹就绪
        
        Args:
            points: 轨迹点列表
            time_step: 每个点之间的时间间隔（秒），用于控制执行速度
        """
        self.current_vision_trajectory = points
        self.trajectory_time_step = time_step  # 保存时间间隔
        
        if time_step:
            frequency = 1.0 / time_step
            self.statusBar.showMessage(f"轨迹生成完成: {len(points)}个点, 时间间隔{time_step*1000:.1f}ms, 频率{frequency:.1f}Hz")
        else:
            self.statusBar.showMessage(f"手动轨迹生成完成，共 {len(points)} 个点位，可以执行或保存")

    def on_execute_vision_trajectory(self):
        if not self.current_vision_trajectory:
            QMessageBox.warning(self, "提醒", "请先加载轨迹数据")
            return
        
        if not self.controller.state.is_enabled:
            QMessageBox.warning(self, "错误", "机器人未使能，无法执行运动")
            return

        self.actual_trajectory_log = []
        self.robot_view.renderer.clear_actual_path()
        self.is_recording_actual = True
            
        if QMessageBox.question(self, "确认执行", f"即将以 {self.playback_frequency}Hz 速率执行轨迹，是否开始?") == QMessageBox.Yes:
            threading.Thread(target=self._run_points_sequence, args=(self.current_vision_trajectory,), daemon=True).start()
        else:
            self.is_recording_actual = False

    def on_execute_vision_trajectory_udp(self):
        """UDP 模式轨迹执行 (自动对位 + 增量随动)"""
        if not self.current_vision_trajectory:
            QMessageBox.warning(self, "提醒", "请先加载轨迹数据")
            return
        
        if not self.controller.state.is_enabled:
            QMessageBox.warning(self, "错误", "机器人未使能")
            return

        self.actual_trajectory_log = []
        self.robot_view.renderer.clear_actual_path()
        
        msg = "系统将执行以下操作：\n1. 自动移动到轨迹起点 (IK)\n2. 自动开启跟随模式\n3. 执行 UDP 增量轨迹\n\n是否开始？"
        if QMessageBox.question(self, "确认执行", msg) == QMessageBox.Yes:
            # 在后台线程中执行流水线
            threading.Thread(target=self._run_udp_pipeline, args=(self.current_vision_trajectory,), daemon=True).start()

    def _run_udp_pipeline(self, points):
        """UDP 自动化流水线：对位 -> 切换 -> 随动"""
        try:
            self.signals.status_updated.emit("步骤 1/3: 正在移动到轨迹起点...")
            p0 = points[0]
            
            # 1. 自动对位 (使用 IK)
            pos_m = [p0[0]/1000.0, p0[1]/1000.0, p0[2]/1000.0]
            q_rad, _ = self.fast_ik.solve_ik(pos=pos_m, rpy_deg=p0[3:], current_joints=np.deg2rad(self.controller.current_joints))
            
            if q_rad:
                # 移动并等待完成 (使用 WebSocket)
                self.controller.move_joint(np.rad2deg(q_rad).tolist(), vels=[30,30,30], wait_for_finish=True)
                time.sleep(0.5)
            else:
                self.signals.error_occurred.emit("无法找到起点逆解，对位失败")
                return

            # 2. 自动启动跟随模式
            self.signals.status_updated.emit("步骤 2/3: 正在初始化跟随模式...")
            ip = self.conn_panel.get_ip()
            # 注意：这里调用 controller 的一键启动逻辑
            if not self.controller.start_follower_mode(ip):
                self.signals.error_occurred.emit("跟随模式启动失败")
                return
            
            # 自动连接 UDP (如果未连接)
            if not self.controller.udp_connected:
                if not self.controller.connect_udp():
                    self.signals.error_occurred.emit("UDP 连接失败")
                    return
            
            time.sleep(1.0) # 等待模式切换稳定

            # 3. 开始 UDP 增量执行
            self.signals.status_updated.emit("步骤 3/3: 开始执行增量轨迹")
            self.is_recording_actual = True
            # 使用轨迹自带的时间间隔（如果有的话）
            time_step = getattr(self, 'trajectory_time_step', None)
            self._run_points_sequence_udp(points, time_step)
            
        except Exception as e:
            self.signals.error_occurred.emit(f"流水线执行异常: {e}")

    def _run_points_sequence_udp(self, points, time_step=None):
        """UDP 增量模式执行逻辑
        
        Args:
            points: 轨迹点列表
            time_step: 每个点之间的时间间隔（秒），如果为None则使用默认频率
        """
        # 确定时间间隔
        if time_step is not None:
            target_interval = time_step
            frequency = 1.0 / time_step
            print(f"\nUDP 轨迹执行 (频率: {frequency:.1f}Hz, 间隔: {time_step*1000:.2f}ms)")
        else:
            target_interval = 1.0 / self.playback_frequency
            frequency = self.playback_frequency
            print(f"\nUDP 轨迹执行 (Hz: {self.playback_frequency})")
        
        try:
            # follower_cart 启动后，机器人当前位置即为"零点"
            # pe 发送的是相对于零点的累积偏移 (m, rad)
            # 这里以 CSV 第一个点为基准，后续点相对于它计算偏移
            # 前提：_run_udp_pipeline 已通过 IK 将机器人移到 CSV 第一个点
            p_start = np.array(points[0])

            # 重置 controller 的累积偏移，保持与手动遥操作一致
            self.controller.follower_offset.fill(0)

            print(f"[UDP轨迹] CSV起始点 p_start = {np.round(p_start, 3).tolist()}")
            print(f"[UDP轨迹] 共 {len(points)} 个点，开始执行...")
            print(f"{'帧':<6} | {'CSV_X':>10} {'CSV_Y':>10} {'CSV_Z':>10} | "
                  f"{'delta_X(mm)':>12} {'delta_Y(mm)':>12} {'delta_Z(mm)':>12} | "
                  f"{'发送_x(m)':>10} {'发送_y(m)':>10} {'发送_z(m)':>10} | "
                  f"{'机器人X':>10} {'机器人Y':>10} {'机器人Z':>10}")
            print("-" * 130)

            for i, p in enumerate(points):
                t_loop_start = time.perf_counter()
                if not self.controller.state.is_enabled or not self.controller.is_follower_mode:
                    break

                self.current_target_point = p

                # 当前点相对于起始点的偏移 (mm, deg)
                delta = np.array(p) - p_start

                # 转换单位: mm -> m, deg -> rad，Z轴取反与 send_raw_increment 保持一致
                dx, dy, dz = delta[:3] / 1000.0
                drx, dry, drz = np.deg2rad(delta[3:])

                # 坐标映射：CSV基座坐标 → 工具坐标系 (X/Y互换，Z取反)
                self.controller.follower_offset[:] = [dy, dx, -dz, dry, drx, drz]

                # 发送给机器人
                self.controller.udp_client.send_pose_euler(dy, dx, -dz, dry, drx, drz)

                # 每帧打印
                tcp = self.controller.state.get_tcp()
                tcp_str = f"{tcp[0]:>10.2f} {tcp[1]:>10.2f} {tcp[2]:>10.2f}" if tcp else f"{'N/A':>10} {'N/A':>10} {'N/A':>10}"
                print(f"{i:<6} | {p[0]:>10.2f} {p[1]:>10.2f} {p[2]:>10.2f} | "
                      f"{delta[0]:>12.2f} {delta[1]:>12.2f} {delta[2]:>12.2f} | "
                      f"{dy:>10.4f} {dx:>10.4f} {-dz:>10.4f} | "
                      f"{tcp_str}")

                time.sleep(max(0, target_interval - (time.perf_counter() - t_loop_start)))
            
            # 轨迹播放完毕，持续发送最终目标 2 秒，让机器人到位
            print(f"[UDP轨迹] 轨迹播完，保持最终目标 2s 等待机器人到位...")
            hold_start = time.perf_counter()
            while time.perf_counter() - hold_start < 2.0:
                self.controller.udp_client.send_pose_euler(
                    *self.controller.follower_offset[:3],
                    *self.controller.follower_offset[3:]
                )
                time.sleep(0.02)

            tcp = self.controller.state.get_tcp()
            if tcp:
                print(f"[UDP轨迹] 最终位置: X={tcp[0]:.2f} Y={tcp[1]:.2f} Z={tcp[2]:.2f}")

            self.is_recording_actual = False
            self.signals.status_updated.emit("UDP 轨迹执行完成")
        except Exception as e:
            self.signals.error_occurred.emit(f"UDP 执行异常: {e}")
        finally:
            self.signals.status_updated.emit("正在自动关闭跟随模式...")
            self.controller.cmd_stop_follower()

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

    def on_command_finished(self, success, message): self.update_status(message)
    def on_connection_changed(self, connected, port_type): 
        self.statusBar.showMessage(f"{port_type} 已{'连接' if connected else '断开'}")

    def closeEvent(self, event):
        self.view_timer.stop()
        self.controller.stop()
        self.fast_ik.close()
        event.accept()
