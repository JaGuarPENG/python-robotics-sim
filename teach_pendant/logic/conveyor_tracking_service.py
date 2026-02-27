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
        self.hover_count = 0      # [新增] 悬停稳定计数器
        
        # === 运动控制参数 ===
        self.tool_length = 0.20         # [新增] 20cm 探针长度 (m)
        self.hover_height = 0.10        # m (探针尖端距离小球高度)
        self.target_z_surface = 0.211   # m (小球表面真实高度Z)
        self.current_z_target = self.target_z_surface + self.hover_height  # 当前探针尖端目标高度
        
        self.approach_speed_z = 0.05    # Z轴逼近/复位速度: 0.05m/s (50mm/s)
        self.conveyor_speed_y = 0.05    # [优化] 传送带默认速度: 0.05m/s
        self.xy_threshold = 0.010       # [优化] 水平误差阈值 10mm
        
        # === 动态前馈参数 ===
        self.look_ahead_frames = 1.5    # [优化] 增加预瞄帧数至 1.5，补偿下压过程中的动态滞后
        self.intercept_lead_y = 0.0     # [优化] 彻底移除人工提前量，直接瞄准球心
        
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
                    
                    # 获取实际高度用于辅助判定
                    tcp_mm_cur = self.controller.state.get_tcp()
                    actual_tip_z_cur = (tcp_mm_cur[2] / 1000.0) - self.tool_length
                    
                    # 判定条件：水平对准且高度已降到悬停位附近 (放宽至 70mm 以兼容 50mm 坐标偏置)
                    if real_xy_distance < self.xy_threshold and abs(actual_tip_z_cur - self.current_z_target) < 0.070:
                        self.hover_count += 1
                    else:
                        self.hover_count = 0
                    
                    # 必须连续 5 帧稳定对准才开始下压 (约 150ms)
                    if self.hover_count >= 5:
                        print(f"--- [状态机] 目标已稳定截获 (计数={self.hover_count})，进入 APPROACHING ---")
                        self.state = "APPROACHING"
                        self.hover_count = 0

                elif self.state == "APPROACHING":
                    # 递减目标高度，稍微下穿表面以抵消机械静差
                    self.current_z_target -= self.approach_speed_z * self.loop_interval
                    if self.current_z_target < self.target_z_surface - 0.010:
                        self.current_z_target = self.target_z_surface - 0.010
                    
                    # 获取实际探针尖端高度 (m)
                    tcp_mm = self.controller.state.get_tcp()
                    actual_tip_z = (tcp_mm[2] / 1000.0) - self.tool_length
                    
                    # 【核心修正】判定门槛增加 50mm 补偿
                    # 视觉表面在 0.211，但机器人物理反馈在 0.261 左右即为触碰
                    if actual_tip_z <= 0.265:
                        # 触碰瞬间，重新获取最实时的球位置进行对比
                        latest_target_pos, _ = self.robot_view.renderer.get_tracking_target()
                        if latest_target_pos is None: latest_target_pos = target_pos

                        final_err_x = (tcp_mm[0]/1000.0) - latest_target_pos[0]
                        final_err_y = (tcp_mm[1]/1000.0) - latest_target_pos[1]
                        final_dist_mm = math.sqrt(final_err_x**2 + final_err_y**2) * 1000.0
                        
                        print(f"--- [状态机] 已触碰小球表面，最终误差: {final_dist_mm:.2f}mm (X:{final_err_x*1000:.1f}, Y:{final_err_y*1000:.1f}) ---")
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
                        
                        # 仅在悬停和逼近阶段打印实时误差，复位离开阶段不打印
                        if self.state in ["HOVERING", "APPROACHING"] and int(loop_start_time * 10) % 10 == 0: 
                            tcp_mm = self.controller.state.get_tcp()
                            actual_tip_z = (tcp_mm[2] / 1000.0) - self.tool_length
                            print(f"[{self.state}] 目标Z:{self.current_z_target:.3f}m | 实际尖端Z:{actual_tip_z:.3f}m | XY误差:{real_xy_distance*1000.0:.1f}mm")
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
