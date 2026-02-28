"""
传送带动态追踪与触碰服务 (最终版: 4阶段状态机 + 实机下发 + 回家逻辑)
负责：在 30Hz 频率下实现 发现->悬停(带前馈补偿)->逼近->复位的状态流转，并支持无目标时返回初始位置。
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
        self.tracking_thread = None
        self.loop_hz = 60.0  # 提升频率至 60Hz
        self.loop_interval = 1.0 / self.loop_hz
        
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
        
        # === 动态前馈参数 ===
        self.look_ahead_frames = 3.0    # 60Hz 下预瞄 3 帧约 50ms
        self.system_latency_offset = 0.02 # [优化] 系统的隐性延迟系数 (经由 4.5mm 误差反推，0.11 - 0.09 = 0.02秒)
        
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
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        self.status_updated.emit("传送带追踪服务已启动")
        print("\n[阶段四] 传送带追踪服务已启动，处于 OBSERVING 状态")

    def stop_tracking(self):
        if not self.is_running:
            return
            
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        self.status_updated.emit("传送带追踪服务已停止")
        print("[阶段四] 传送带追踪服务已停止")

    def _tracking_loop(self):
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
                # 总前馈时间 = (1/60 * 3) + 0.2 = 0.05 + 0.2 = 0.25 秒
                total_latency_seconds = (self.loop_interval * self.look_ahead_frames) + self.system_latency_offset
                predicted_target_y = target_pos[1] + (self.conveyor_speed_y * total_latency_seconds)
                
                # 获取真实 TCP 用于计算误差 (单位转换: mm -> m)
                tcp_mm = self.controller.state.get_tcp()
                tcp_m = [tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0]
                
                # 计算当前末端相对于小球实际位置的水平误差 (用于判定是否截获)
                # 注意：这是真实的物理误差，不是相对盲抓理论点的误差
                error_x = tcp_m[0] - target_pos[0]
                error_y = tcp_m[1] - target_pos[1]
                real_xy_distance = math.sqrt(error_x**2 + error_y**2)

                # ==========================
                # 状态机核心控制流
                # ==========================
                if self.state == "HOVERING":
                    self.current_z_target = target_pos[2] + self.hover_height
                    
                    # 获取实际高度用于辅助判定
                    tcp_mm_cur = self.controller.state.get_tcp()
                    actual_tip_z_cur = (tcp_mm_cur[2] / 1000.0) - self.tool_length
                    
                    # [新增] 双轨下压触发机制
                    triggered_approach = False
                    
                    # 判定条件：水平对准且高度已降到悬停位附近 (放宽至 70mm 以兼容 50mm 坐标偏置)
                    if real_xy_distance < self.xy_threshold and abs(actual_tip_z_cur - self.current_z_target) < 0.070:
                        self.hover_count += 1
                    else:
                        self.hover_count = 0
                    
                    # 必须连续 5 帧稳定对准才开始下压 (约 150ms)
                    if self.hover_count >= 5:
                        print(f"--- [状态机] 目标已稳定截获 (计数={self.hover_count})，进入 APPROACHING ---")
                        triggered_approach = True

                    if triggered_approach:
                        self.state = "APPROACHING"
                        self.hover_count = 0
                        if self.use_blind_tracking:
                            # [核心优化] 一旦对准，锁定当前的 X 和预测的 Y，进入纯物理开环跟随
                            self.locked_x = target_pos[0]
                            self.locked_y = predicted_target_y
                            print(f"--- [盲抓模式] 已锁定坐标 X:{self.locked_x:.3f}, Y:{self.locked_y:.3f} ---")

                elif self.state == "APPROACHING":
                    # [核心优化] 不再依赖视觉，根据产线已知速度，在锁定的 Y 上做纯数学累加 (盲抓)
                    self.locked_y += self.conveyor_speed_y * self.loop_interval
                    
                    # [新增计算] 动态下压速度与时间绑定
                    # 我们希望下压过程(下降 hover_height)在一个固定的安全时间窗口内完成，例如 0.5 秒
                    # 这样速度会随着您设置的 hover_height 动态变化
                    approach_time_budget = 0.5 # 秒
                    dynamic_approach_speed_z = self.hover_height / approach_time_budget
                    
                    # 递减目标高度，稍微下穿表面以抵消机械静差
                    self.current_z_target -= dynamic_approach_speed_z * self.loop_interval
                    if self.current_z_target < self.target_z_surface - 0.010:
                        self.current_z_target = self.target_z_surface - 0.010
                    
                    # 获取实际探针尖端高度 (m)
                    tcp_mm = self.controller.state.get_tcp()
                    actual_tip_z = (tcp_mm[2] / 1000.0) - self.tool_length
                    
                    # 【核心修正】判定门槛增加 50mm 补偿
                    # 视觉表面在 0.211，但机器人物理反馈在 0.261 左右即为触碰
                    if actual_tip_z <= 0.265:
                        # 触碰瞬间，计算最终相对于真实小球的物理误差
                        final_err_x = (tcp_mm[0]/1000.0) - target_pos[0]
                        final_err_y = (tcp_mm[1]/1000.0) - target_pos[1]
                        final_dist_mm = math.sqrt(final_err_x**2 + final_err_y**2) * 1000.0
                        
                        print(f"--- [状态机] 已触碰小球表面，最终物理误差: {final_dist_mm:.2f}mm (X:{final_err_x*1000:.1f}, Y:{final_err_y*1000:.1f}) ---")
                        
                        # [动态自适应校准 - Auto Calibration]
                        # 误差 = 机械臂当前Y - 真实小球Y
                        # 假设机械臂完美走到了预测的锁死点，那么:
                        # err_y = (真实物理延迟 - 我们的预判延迟) * 速度
                        # 如果 err_y > 0 (超前)，说明我们的预判延迟给大了，需要减小 offset
                        if abs(self.conveyor_speed_y) > 0.001:
                            learning_rate = 0.3  # 学习率，每次只修正 30% 避免单次突变导致震荡
                            time_correction = final_err_y / self.conveyor_speed_y
                            self.system_latency_offset -= time_correction * learning_rate
                            # 限制范围，避免因为偶尔的异常识别导致参数飞掉 (0ms ~ 300ms)
                            self.system_latency_offset = max(0.0, min(0.3, self.system_latency_offset))
                            print(f"--- [自适应校准] 自动修正延迟系数 -> 新 offset: {self.system_latency_offset:.4f}s ---")
                            
                        self.state = "RETURNING"
                        self.robot_view.renderer.mark_target_reached()

                elif self.state == "RETURNING":
                    # 复位期间继续保持 Y 轴速度同步，防止斜拉扯
                    self.locked_y += self.conveyor_speed_y * self.loop_interval
                    
                    self.current_z_target += (self.approach_speed_z * 2) * self.loop_interval
                    if self.current_z_target >= target_pos[2] + self.hover_height:
                        self.current_z_target = target_pos[2] + self.hover_height
                        print(f"--- [状态机] 复位完成，回到 OBSERVING 状态 ---")
                        self.state = "OBSERVING"
                        self.last_target_id = None
                
                # ==========================
                # IK 目标解算与实机指令下发
                # ==========================
                if self.state != "OBSERVING":
                    # 获取当前 J1 角度用于方向锁定，防止 J6 乱转
                    current_joints_deg = self.controller.state.get_joints()
                    current_j1 = current_joints_deg[0]
                    
                    # 最终目标位置: 
                    # 如果是在 APPROACHING 或 RETURNING 阶段，使用纯速度累加的 locked_x / locked_y
                    if self.state in ["APPROACHING", "RETURNING"]:
                        final_target_pos = [self.locked_x, self.locked_y, self.current_z_target + self.tool_length]
                    else:
                        final_target_pos = [target_pos[0], predicted_target_y, self.current_z_target + self.tool_length]
                    
                    # 设定姿态: 根据 Home 点 [180, 0, -90] 修正，加上 J1 补偿实现“随动”
                    hover_rpy_deg = [180.0, 0.0, -90.0 + current_j1] 
                    
                    # === [测时开始] 记录解算与发送的耗时 ===
                    t_ik_send_start = time.perf_counter()
                    
                    target_q_rad, ik_time = self.fast_ik.solve_ik(
                        pos=final_target_pos,
                        rpy_deg=hover_rpy_deg,
                        current_joints=np.deg2rad(current_joints_deg)
                    )
                    
                    if target_q_rad is not None:
                        target_joints_deg = np.rad2deg(target_q_rad).tolist()
                        self.controller.move_joint(target_joints_deg, vels=[100]*6, wait_for_finish=False)
                        
                        # === [测时结束] ===
                        ik_send_latency_ms = (time.perf_counter() - t_ik_send_start) * 1000.0
                        
                        # 仅在悬停和逼近阶段打印实时误差，复位离开阶段不打印
                        if self.state in ["HOVERING", "APPROACHING"] and int(loop_start_time * 10) % 10 == 0: 
                            tcp_mm = self.controller.state.get_tcp()
                            actual_tip_z = (tcp_mm[2] / 1000.0) - self.tool_length
                            print(f"[{self.state}] 目标Z:{self.current_z_target:.3f}m | 实际尖端Z:{actual_tip_z:.3f}m | XY误差:{real_xy_distance*1000.0:.1f}mm | 计算&发送延时:{ik_send_latency_ms:.2f}ms")
                    else:
                        print(f"[{self.state}] 警告: IK解算失败")

            else:
                # --------------------------
                # 目标丢失或任务完成: 回到初始位置
                # --------------------------
                if self.state != "OBSERVING":
                    print(f"--- [状态机] 目标丢失，返回初始位置 ---")
                    self.state = "OBSERVING"
                    self.last_target_id = None
                
                # 在无目标期间，持续平滑地回到快速定位1点
                self.controller.move_joint(self.home_joints, vels=[60]*6, wait_for_finish=False)
            
            # 5. 频率控制
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.loop_interval - elapsed)
            time.sleep(sleep_time)
