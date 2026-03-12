"""
传送带动态追踪与触碰服务 (双频架构: 60Hz视觉 + 125Hz控制)
负责：视觉线程以60Hz运行状态机，控制线程以125Hz发送关节指令到工控机
"""

import threading
import time
import math
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .tracking_algorithms import PIDTrackingAlgorithm, SimTrackingAlgorithm, OffsetTrackingAlgorithm

class ConveyorTrackingService(QObject):
    status_updated = pyqtSignal(str)
    
    def __init__(self, controller, fast_ik, robot_view):
        super().__init__()
        self.controller = controller
        self.fast_ik = fast_ik
        self.robot_view = robot_view
        
        self.is_running = False
        
        # === 双频架构 ===
        # 视觉线程：60Hz，负责目标检测和状态机
        self.vision_thread = None
        self.vision_hz = 60.0
        self.vision_interval = 1.0 / self.vision_hz
        
        # 控制线程：125Hz，负责插值、IK和发送关节角
        self.control_thread = None
        self.control_hz = 125.0
        self.control_interval = 1.0 / self.control_hz
        
        # === 线程安全的目标位置缓冲区 ===
        self._target_lock = threading.Lock()
        self._target_buffer = {
            'pos': None,           # [x, y, z] 目标TCP位置（米）
            'rpy': None,           # [rx, ry, rz] 目标姿态（度）
            'timestamp': 0.0,      # 写入时间戳
            'valid': False,        # 数据是否有效
            'state': 'OBSERVING',  # 当前状态
            'has_target': False,   # 是否检测到目标
            'conveyor_speed': 0.0, # 传送带速度（用于插值）
        }
        
        # === 状态机变量 ===
        self.last_target_id = None
        self.state = "OBSERVING"  # OBSERVING, HOVERING, APPROACHING, RETURNING
        self.hover_count = 0      # [新增] 悬停稳定计数器
        
        # === UDP follower_cart 追踪模式 ===
        self.use_udp_follower = False
        self.p0 = None   # follower_cart 零点 (m, 基座坐标系)

        # === 追踪模式控制 ===
        self.use_blind_tracking = True # [新增] 是否使用盲抓(开环)下压模式
        self.locked_x = 0.0            # 盲抓锁定的X坐标
        self.locked_y = 0.0            # 盲抓锁定的Y坐标
        
        # === 运动控制参数 ===
        self.tool_length = 0.20         # [新增] 20cm 探针长度 (m)
        self.hover_height = 0.025        # m (探针尖端距离小球高度)
        self.target_z_surface = 0.211   # m (小球表面真实高度Z)
        self.current_z_target = self.target_z_surface + self.hover_height  # 当前探针尖端目标高度
        
        self.approach_speed_z = 0.05    # Z轴逼近/复位速度: 0.05m/s (50mm/s)
        self.conveyor_speed_y = 0.05    # [优化] 传送带默认速度: 0.05m/s
        self.xy_threshold = 0.001       # 严格要求1mm以内触发
        self.blind_xy_threshold = 0.001 # 严格要求1mm以内触发
        
        # === 动态前馈参数 (适配125Hz控制频率) ===
        # 125Hz下预瞄6帧约48ms，与60Hz下3帧约50ms等效
        self.look_ahead_frames = 6.0    # 125Hz 控制周期下的预瞄帧数

        # === PID 控制器参数 (Y轴跟踪) ===
        self.pi_kp = 0.2  # 比例系数: 提高响应速度，快速逼近
        self.pi_ki = 0.05 # 积分系数: 消除1mm以内的微小稳态误差
        self.pi_kd = 0.03 # 微分系数: 增加阻尼，防止超调震荡，帮助快速稳定在1mm内
        self.pi_integral_y = 0.0 # Y轴误差积分累计
        self.pi_last_error_y = 0.0 # 上一帧的误差
        self.pi_compensation_y = 0.0 # 最终PID计算出的Y轴补偿量

        # === 仿真算法 (速度前馈+限幅) 专用参数 ===
        self.use_sim_algorithm = False
        self.sim_kp = np.array([2.0, 2.0, 2.0])  # XYZ三轴P参数
        self.sim_ki = np.array([0.5, 0.5, 0.5])
        self.sim_max_step = 0.02 # 单控制周期(约0.016s)最大移动距离 20mm
        self.sim_integral_limit = 0.1
        self.sim_integral_error = np.zeros(3)
        self.sim_virtual_pos = None # 虚拟当前位置 (用于平滑积分)

        # === 隐性 Offset 追踪算法专用参数 ===
        self.use_dynamic_offset = False
        self.system_latency_offset = 0.12 # 系统的初始隐性延迟系数 (约20ms)

        # === 追踪算法策略类 ===
        self.algorithms = {
            "PID": PIDTrackingAlgorithm(self),
            "SIM": SimTrackingAlgorithm(self),
            "OFFSET": OffsetTrackingAlgorithm(self)
        }
        self.active_algorithm = self.algorithms["PID"]

        # === 初始位置 (快速定位1) ===
        self.home_joints = [0, -15, 105, 0, -90, 0] # 角度

    def set_conveyor_speed(self, speed):
        """动态修改追踪预测速度"""
        self.conveyor_speed_y = speed
        print(f"[TrackingService] 追踪预测速度已更新为: {speed} m/s")

    def start_tracking(self):
        if self.is_running:
            return

        if self.use_udp_follower:
            if not self.controller.udp_connected:
                self.controller.connect_udp(9998)
            self.p0 = self.controller.init_udp_tracking_mode()
            print(f"[UDP追踪] 零点P0 = {[f'{v:.4f}' for v in self.p0]}")

        self.is_running = True
        self.state = "OBSERVING"
        self.last_target_id = None
        self.sim_integral_error = np.zeros(3)
        self.sim_virtual_pos = None

        # 启动双线程
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.vision_thread.start()
        self.control_thread.start()

        self.status_updated.emit("传送带追踪服务已启动 (60Hz视觉+125Hz控制)")
        print("\n[阶段四] 传送带追踪服务已启动")
        print(f"  - 视觉线程: {self.vision_hz}Hz (目标检测+状态机)")
        print(f"  - 控制线程: {self.control_hz}Hz (插值+IK+发送)")

    def stop_tracking(self):
        if not self.is_running:
            return

        self.is_running = False
        if self.vision_thread:
            self.vision_thread.join(timeout=1.0)
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        if self.use_udp_follower:
            self.controller.cmd_stop_follower()
            self.p0 = None

        self.status_updated.emit("传送带追踪服务已停止")
        print("[阶段四] 传送带追踪服务已停止")

    def _vision_loop(self):
        while self.is_running:
            loop_start_time = time.perf_counter()
            
            # 1. 尝试获取当前追踪目标
            target_pos, target_id = None, None
            if hasattr(self.robot_view, 'renderer') and self.robot_view.renderer:
                target_pos, target_id = self.robot_view.renderer.get_tracking_target()
                
            if target_pos is not None and target_id is not None:
                # 获取真实 TCP 用于计算误差 (单位转换: mm -> m)
                tcp_mm = self.controller.state.get_tcp()
                tcp_m = [tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0]
                
                # 选择激活的算法
                if self.use_sim_algorithm:
                    alg = self.algorithms["SIM"]
                elif self.use_dynamic_offset:
                    alg = self.algorithms["OFFSET"]
                else:
                    alg = self.algorithms["PID"]
                    
                if self.active_algorithm != alg:
                    self.active_algorithm = alg
                    self.active_algorithm.reset()
                    
                # 执行策略
                final_target_pos = self.active_algorithm.execute_vision_cycle(target_pos, target_id, tcp_m, loop_start_time)
                
                # 写入缓冲区供控制线程读取
                if final_target_pos is not None:
                    with self._target_lock:
                        self._target_buffer['pos'] = final_target_pos
                        self._target_buffer['timestamp'] = time.perf_counter()
                        self._target_buffer['valid'] = True
                        self._target_buffer['state'] = self.state
                        self._target_buffer['has_target'] = True
                        self._target_buffer['conveyor_speed'] = self.conveyor_speed_y

            else:
                if self.state != "OBSERVING":
                    if self.use_udp_follower:
                        print(f"--- [UDP纯追踪] 目标丢失，原地悬停等待 ---")
                    else:
                        print(f"--- [状态机] 目标丢失，返回初始位置 ---")
                    self.state = "OBSERVING"
                    self.last_target_id = None
                    self.active_algorithm.reset()
                
                with self._target_lock:
                    self._target_buffer['valid'] = False
                    self._target_buffer['state'] = "OBSERVING"
                    self._target_buffer['has_target'] = False

            # 频率控制 (60Hz)
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.vision_interval - elapsed)
            time.sleep(sleep_time)

    def _control_loop(self):
        while self.is_running:
            loop_start_time = time.perf_counter()
            
            with self._target_lock:
                target_data = self._target_buffer.copy()

            state = target_data.get('state', 'OBSERVING')
            has_target = target_data.get('has_target', False)
            valid = target_data.get('valid', False)
            target_pos = target_data.get('pos')
            
            if self.use_udp_follower and self.p0 is not None:
                # UDP追踪悬停高度（小球上方50mm）
                hover_height = 0.050  # 50mm
                # 自适应前馈参数：预瞄时间（秒）
                lookahead_time = 0.15  # 150ms预瞄，可根据实际情况调整
                
                if has_target and valid and target_pos is not None and state != "OBSERVING":
                    # 计算视觉延迟
                    time_since_vision = loop_start_time - target_data['timestamp']
                    
                    # 计算前馈补偿后的目标位置
                    # 传送带沿Y轴运动，速度为conveyor_speed_y
                    conveyor_speed = target_data.get('conveyor_speed', self.conveyor_speed_y)
                    
                    # 预瞄位置 = 当前位置 + 速度 × (预瞄时间 + 视觉延迟)
                    total_latency = lookahead_time + time_since_vision
                    predicted_pos = list(target_pos)
                    predicted_pos[1] += conveyor_speed * total_latency  # Y轴前馈补偿
                    
                    # 计算悬停位置：小球上方50mm
                    hover_pos = list(predicted_pos)
                    hover_pos[2] += hover_height  # Z轴向上偏移50mm
                    
                    # 发送UDP偏移到悬停位置（带前馈补偿）
                    self.controller.send_udp_target(hover_pos, self.p0)
                    self._last_udp_target = hover_pos
                    
                    # 调试输出（每30帧输出一次，避免刷屏）
                    if not hasattr(self, '_udp_debug_counter'):
                        self._udp_debug_counter = 0
                    self._udp_debug_counter += 1
                    if self._udp_debug_counter % 30 == 0:
                        print(f"[UDP追踪] 当前Y:{target_pos[1]:.3f}m 预测Y:{predicted_pos[1]:.3f}m "
                              f"速度:{conveyor_speed:.3f}m/s 延迟:{total_latency*1000:.1f}ms")
                else:
                    # 目标丢失：悬停在最后看到小球的坐标上方，不回原点
                    if hasattr(self, '_last_udp_target') and self._last_udp_target is not None:
                        self.controller.send_udp_target(self._last_udp_target, self.p0)
            else:
                if has_target and valid and target_pos is not None and state != "OBSERVING":
                    current_joints_deg = self.controller.state.get_joints()
                    current_j1 = current_joints_deg[0]

                    # 线性插值，使125Hz下运行平滑
                    time_since_vision = loop_start_time - target_data['timestamp']
                    interpolated_pos = list(target_pos)
                    if time_since_vision < 0.05:  # 只在延迟合理的情况下插值
                        interpolated_pos[1] += target_data['conveyor_speed'] * time_since_vision

                    hover_rpy_deg = [180.0, 0.0, -90.0 + current_j1]

                    target_q_rad, ik_time = self.fast_ik.solve_ik(
                        pos=interpolated_pos,
                        rpy_deg=hover_rpy_deg,
                        current_joints=np.deg2rad(current_joints_deg)
                    )

                    if target_q_rad is not None:
                        target_joints_deg = np.rad2deg(target_q_rad).tolist()
                        self.controller.move_joint(target_joints_deg, vels=[100]*6, wait_for_finish=False)
                    else:
                        print(f"[{state}] 警告: IK解算失败")

                else:
                    # 目标丢失或任务完成: 回到初始位置
                    self.controller.move_joint(self.home_joints, vels=[60]*6, wait_for_finish=False)
            
            # 频率控制 (125Hz)
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.control_interval - elapsed)
            time.sleep(sleep_time)
