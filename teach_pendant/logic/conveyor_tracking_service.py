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
        self.loop_hz = 30.0
        self.loop_interval = 1.0 / self.loop_hz
        
        # === 状态机变量 ===
        self.last_target_id = None
        self.state = "OBSERVING"  # OBSERVING, HOVERING, APPROACHING, RETURNING
        
        # === 运动控制参数 ===
        self.tool_length = 0.20         # [新增] 20cm 探针长度 (m)
        self.hover_height = 0.20        # m (探针尖端距离小球高度)
        self.target_z_surface = 0.211   # m (小球表面高度Z)
        self.current_z_target = self.target_z_surface + self.hover_height  # 当前探针尖端目标高度
        
        self.approach_speed_z = 0.05    # Z轴逼近/复位速度: 0.05m/s (50mm/s)
        self.conveyor_speed_y = 0.05    # [优化] 传送带默认速度: 0.05m/s
        self.xy_threshold = 0.04        # [优化] 水平误差阈值放宽至 4cm，增加锁定成功率
        
        # === 动态前馈参数 ===
        self.look_ahead_frames = 3.0    # [优化] 预瞄 3 帧 (约 100ms) 补偿系统延迟
        self.intercept_lead_y = 0.02    # [优化] Y轴额外增加 2cm 截击提前量
        
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
                    self.current_z_target = target_pos[2] + self.hover_height
                    print(f"\n--- [状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")

                # ==========================
                # 前馈速度补偿与截击 (Feedforward & Intercept)
                # ==========================
                # [优化] 预估小球 Look-ahead 帧后的位置，并额外增加截击提前量，解决系统滞后
                predicted_target_y = target_pos[1] + (self.conveyor_speed_y * self.loop_interval * self.look_ahead_frames) + self.intercept_lead_y
                
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
                    # [优化] 只要距离在 4cm 以内，即刻开始下压
                    if real_xy_distance < self.xy_threshold:
                        print(f"--- [状态机] 目标已截获 (误差 {real_xy_distance*100:.1f}cm < 4cm)，进入 APPROACHING ---")
                        self.state = "APPROACHING"

                elif self.state == "APPROACHING":
                    self.current_z_target -= self.approach_speed_z * self.loop_interval
                    if self.current_z_target <= self.target_z_surface:
                        self.current_z_target = self.target_z_surface
                        print(f"--- [状态机] 已触碰小球表面，进入 RETURNING (复位) ---")
                        self.state = "RETURNING"
                        self.robot_view.renderer.mark_target_reached()

                elif self.state == "RETURNING":
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
                    # XY 包含前馈预测
                    # Z = 探针尖端高度 + 探针长度 (补偿到法兰盘中心)
                    final_target_pos = [target_pos[0], predicted_target_y, self.current_z_target + self.tool_length]
                    
                    # 设定姿态: 根据 Home 点 [180, 0, -90] 修正，加上 J1 补偿实现“随动”
                    hover_rpy_deg = [180.0, 0.0, -90.0 + current_j1] 
                    
                    target_q_rad, ik_time = self.fast_ik.solve_ik(
                        pos=final_target_pos,
                        rpy_deg=hover_rpy_deg,
                        current_joints=np.deg2rad(current_joints_deg)
                    )
                    
                    if target_q_rad is not None:
                        target_joints_deg = np.rad2deg(target_q_rad).tolist()
                        self.controller.move_joint(target_joints_deg, vels=[100]*6, wait_for_finish=False)
                        
                        if int(loop_start_time * 10) % 10 == 0: 
                            print(f"[{self.state}] 探针尖端Z:{self.current_z_target:.3f}m | 实际XY误差:{real_xy_distance*100:.1f}cm")
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
