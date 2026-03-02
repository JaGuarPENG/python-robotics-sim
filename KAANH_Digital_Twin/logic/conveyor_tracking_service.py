"""
传送带动态追踪与触碰服务 (双频架构: 60Hz视觉 + 125Hz控制)
负责：视觉线程以60Hz运行状态机，控制线程以125Hz发送关节指令到工控机
"""

import threading
import time
import math
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

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
        self.xy_threshold = 0.001       # [优化] 水平误差阈值 3mm
        self.blind_xy_threshold = 0.001 # [新增] 盲抓触发的极小误差阈值 (如5mm以内直接触发)
        
        # === 动态前馈参数 (适配125Hz控制频率) ===
        # 125Hz下预瞄6帧约48ms，与60Hz下3帧约50ms等效
        self.look_ahead_frames = 6.0    # 125Hz 控制周期下的预瞄帧数
        self.system_latency_offset = 0.02 # 系统的隐性延迟系数
        
        # === 初始位置 (快速定位1) ===
        self.home_joints = [0, -15, 105, 0, -90, 0] # 角度

    def set_conveyor_speed(self, speed):
        """动态修改追踪预测速度"""
        self.conveyor_speed_y = speed
        print(f"[TrackingService] 追踪预测速度已更新为: {speed} m/s")

    def start_tracking(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.state = "OBSERVING"
        self.last_target_id = None
        
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
                # 如果是新进来的目标，立刻重置为 HOVERING
                if self.last_target_id != target_id:
                    self.last_target_id = target_id
                    self.state = "HOVERING"
                    self.hover_count = 0  # 重置计数
                    self.current_z_target = target_pos[2] + self.hover_height
                    print(f"\n--- [状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")

                # ==========================
                # 前馈速度补偿与截击 (Feedforward & Intercept)
                # ==========================
                # [优化] 使用纯理论物理公式： 预测位置 = 当前位置 + 速度 * (通信与处理时间 + 机械响应延时)
                total_latency_seconds = (self.control_interval * self.look_ahead_frames) + self.system_latency_offset
                predicted_target_y = target_pos[1] + (self.conveyor_speed_y * total_latency_seconds)
                
                # 获取真实 TCP 用于计算误差 (单位转换: mm -> m)
                tcp_mm = self.controller.state.get_tcp()
                tcp_m = [tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0]
                
                # 计算当前末端相对于小球实际位置的水平误差 (用于判定是否截获)
                error_x = tcp_m[0] - target_pos[0]
                error_y = tcp_m[1] - target_pos[1]
                real_xy_distance = math.sqrt(error_x**2 + error_y**2)

                # ==========================
                # 状态机核心控制流
                # ==========================
                if self.state == "HOVERING":
                    self.current_z_target = target_pos[2] + self.hover_height
                    actual_tip_z_cur = tcp_m[2] - self.tool_length
                    
                    # 判定条件：水平对准且高度已降到悬停位附近
                    if real_xy_distance < self.xy_threshold and abs(actual_tip_z_cur - self.current_z_target) < 0.070:
                        self.hover_count += 1
                    else:
                        self.hover_count = 0
                    
                    if self.hover_count >= 5:
                        print(f"--- [状态机] 目标已稳定截获 (计数={self.hover_count})，进入 APPROACHING ---")
                        self.state = "APPROACHING"
                        self.hover_count = 0
                        if self.use_blind_tracking:
                            self.locked_x = target_pos[0]
                            self.locked_y = predicted_target_y
                            print(f"--- [盲抓模式] 已锁定坐标 X:{self.locked_x:.3f}, Y:{self.locked_y:.3f} ---")

                elif self.state == "APPROACHING":
                    self.locked_y += self.conveyor_speed_y * self.vision_interval
                    
                    approach_time_budget = 0.5 # 秒
                    dynamic_approach_speed_z = self.hover_height / approach_time_budget
                    
                    self.current_z_target -= dynamic_approach_speed_z * self.vision_interval
                    if self.current_z_target < self.target_z_surface - 0.010:
                        self.current_z_target = self.target_z_surface - 0.010
                    
                    actual_tip_z = tcp_m[2] - self.tool_length
                    
                    if actual_tip_z <= 0.265:
                        final_err_x = tcp_m[0] - target_pos[0]
                        final_err_y = tcp_m[1] - target_pos[1]
                        final_dist_mm = math.sqrt(final_err_x**2 + final_err_y**2) * 1000.0
                        
                        print(f"--- [状态机] 已触碰小球表面，最终物理误差: {final_dist_mm:.2f}mm (X:{final_err_x*1000:.1f}, Y:{final_err_y*1000:.1f}) ---")
                        
                        if abs(self.conveyor_speed_y) > 0.001:
                            learning_rate = 0.3
                            time_correction = final_err_y / self.conveyor_speed_y
                            self.system_latency_offset -= time_correction * learning_rate
                            self.system_latency_offset = max(0.0, min(0.3, self.system_latency_offset))
                            print(f"--- [自适应校准] 自动修正延迟系数 -> 新 offset: {self.system_latency_offset:.4f}s ---")
                            
                        self.state = "RETURNING"
                        self.robot_view.renderer.mark_target_reached()

                elif self.state == "RETURNING":
                    self.locked_y += self.conveyor_speed_y * self.vision_interval
                    self.current_z_target += (self.approach_speed_z * 2) * self.vision_interval
                    if self.current_z_target >= target_pos[2] + self.hover_height:
                        self.current_z_target = target_pos[2] + self.hover_height
                        print(f"--- [状态机] 复位完成，回到 OBSERVING 状态 ---")
                        self.state = "OBSERVING"
                        self.last_target_id = None
                
                if self.state in ["APPROACHING", "RETURNING"]:
                    final_target_pos = [self.locked_x, self.locked_y, self.current_z_target + self.tool_length]
                else:
                    final_target_pos = [target_pos[0], predicted_target_y, self.current_z_target + self.tool_length]
                
                # 写入缓冲区供控制线程读取
                with self._target_lock:
                    self._target_buffer['pos'] = final_target_pos
                    self._target_buffer['timestamp'] = time.perf_counter()
                    self._target_buffer['valid'] = True
                    self._target_buffer['state'] = self.state
                    self._target_buffer['has_target'] = True
                    self._target_buffer['conveyor_speed'] = self.conveyor_speed_y
                    
                if self.state in ["HOVERING", "APPROACHING"] and int(loop_start_time * 10) % 10 == 0: 
                    actual_tip_z = tcp_m[2] - self.tool_length
                    print(f"[{self.state}] 目标Z:{self.current_z_target:.3f}m | 实际尖端Z:{actual_tip_z:.3f}m | XY误差:{real_xy_distance*1000.0:.1f}mm")

            else:
                if self.state != "OBSERVING":
                    print(f"--- [状态机] 目标丢失，返回初始位置 ---")
                    self.state = "OBSERVING"
                    self.last_target_id = None
                
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
            
            if has_target and valid and target_pos is not None and state != "OBSERVING":
                current_joints_deg = self.controller.state.get_joints()
                current_j1 = current_joints_deg[0]
                
                # 线性插值，使125Hz下运行平滑
                time_since_vision = loop_start_time - target_data['timestamp']
                interpolated_pos = list(target_pos)
                if time_since_vision < 0.05:  # 只在延迟合理的情况下插值
                    interpolated_pos[1] += target_data['conveyor_speed'] * time_since_vision

                hover_rpy_deg = [180.0, 0.0, -90.0 + current_j1] 
                
                # === [测时开始] ===
                t_ik_send_start = time.perf_counter()
                
                target_q_rad, ik_time = self.fast_ik.solve_ik(
                    pos=interpolated_pos,
                    rpy_deg=hover_rpy_deg,
                    current_joints=np.deg2rad(current_joints_deg)
                )
                
                if target_q_rad is not None:
                    target_joints_deg = np.rad2deg(target_q_rad).tolist()
                    self.controller.move_joint(target_joints_deg, vels=[100]*6, wait_for_finish=False)
                    # ik_send_latency_ms = (time.perf_counter() - t_ik_send_start) * 1000.0
                else:
                    print(f"[{state}] 警告: IK解算失败")

            else:
                # 目标丢失或任务完成: 回到初始位置
                self.controller.move_joint(self.home_joints, vels=[60]*6, wait_for_finish=False)
            
            # 频率控制 (125Hz)
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.control_interval - elapsed)
            time.sleep(sleep_time)
