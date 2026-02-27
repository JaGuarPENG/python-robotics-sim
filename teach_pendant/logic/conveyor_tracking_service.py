"""
传送带动态追踪与触碰服务 (阶段三: 4阶段状态机)
负责：在 30Hz 频率下实现 发现->悬停(带前馈补偿)->逼近->复位的状态流转。
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
        self.hover_height = 0.20        # m (悬停高度20cm)
        self.target_z_surface = 0.211   # m (小球表面高度Z)
        self.current_z_target = self.target_z_surface + self.hover_height
        
        self.approach_speed_z = 0.05    # Z轴逼近/复位速度: 0.05m/s (50mm/s)
        self.conveyor_speed_y = 0.1     # 传送带速度: 0.1m/s (用于前馈补偿)
        self.xy_threshold = 0.02        # 水平误差阈值: 2cm
        
        # 测试用虚拟距离 (因为阶段三不下发实机指令，真实TCP不会移动)
        self._virtual_xy_distance = 0.10

    def start_tracking(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.state = "OBSERVING"
        self.last_target_id = None
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        self.status_updated.emit("传送带追踪服务已启动")
        print("\n[阶段三] 传送带追踪服务已启动 (OBSERVING: 等待目标...)")

    def stop_tracking(self):
        if not self.is_running:
            return
            
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        self.status_updated.emit("传送带追踪服务已停止")
        print("[阶段三] 传送带追踪服务已停止")

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
                    self._virtual_xy_distance = 0.10  # 假设初始误差10cm
                    print(f"\n--- [状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")

                # ==========================
                # 前馈速度补偿 (Feedforward)
                # ==========================
                # 根据传送带0.1m/s的速度，预测在此次控制周期(33ms)中小球已经往前走了多少
                predicted_target_y = target_pos[1] + self.conveyor_speed_y * self.loop_interval
                
                # 获取真实 TCP 用于计算误差 (暂时用于展示逻辑)
                tcp_mm = self.controller.state.get_tcp()
                tcp_m = [tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0]
                error_x = tcp_m[0] - target_pos[0]
                error_y = tcp_m[1] - target_pos[1]
                real_xy_distance = math.sqrt(error_x**2 + error_y**2)
                
                # [阶段三临时测试] 因为不下发指令机械臂没动，real_xy_distance 永远无法缩减到 2cm 以内。
                # 所以在这里我们用一个自减的虚拟距离，强制让状态能流转下去：
                if self.state == "HOVERING":
                    self._virtual_xy_distance -= 0.05 * self.loop_interval # 模拟以5cm/s的速度靠近
                
                # 为了测试，这里取虚拟距离。在阶段四我们会换成 `real_xy_distance`
                test_xy_distance = max(0.0, self._virtual_xy_distance)

                # ==========================
                # 状态机核心控制流
                # ==========================
                if self.state == "HOVERING":
                    # 保持在 Z 轴上方
                    self.current_z_target = target_pos[2] + self.hover_height
                    
                    # 判断误差是否收敛到 2cm 以内
                    if test_xy_distance < self.xy_threshold:
                        print(f"--- [状态机] 目标已锁定 (误差 < 2cm)，进入 APPROACHING (逼近触碰) ---")
                        self.state = "APPROACHING"

                elif self.state == "APPROACHING":
                    # Z 轴开始平滑下降
                    self.current_z_target -= self.approach_speed_z * self.loop_interval
                    
                    # 触底判定
                    if self.current_z_target <= self.target_z_surface:
                        self.current_z_target = self.target_z_surface
                        print(f"--- [状态机] 已触碰小球表面，进入 RETURNING (复位) ---")
                        self.state = "RETURNING"
                        # 触发渲染器小球变红
                        self.robot_view.renderer.mark_target_reached()

                elif self.state == "RETURNING":
                    # 快速抬升 Z 轴回到悬停点
                    self.current_z_target += (self.approach_speed_z * 2) * self.loop_interval
                    
                    if self.current_z_target >= target_pos[2] + self.hover_height:
                        self.current_z_target = target_pos[2] + self.hover_height
                        print(f"--- [状态机] 复位完成，进入 OBSERVING (等待下一次目标) ---")
                        self.state = "OBSERVING"
                        self.last_target_id = None # 丢弃当前已经变红的目标，等待下一个白球变绿
                
                # ==========================
                # IK 目标解算与验证输出
                # ==========================
                if self.state != "OBSERVING":
                    hover_pos = [target_pos[0], predicted_target_y, self.current_z_target]
                    hover_rpy_deg = [180.0, 0.0, 0.0]
                    
                    target_q_rad, ik_time = self.fast_ik.solve_ik(
                        pos=hover_pos,
                        rpy_deg=hover_rpy_deg,
                        current_joints=np.deg2rad(self.controller.state.get_joints())
                    )
                    
                    if target_q_rad is not None:
                        target_joints_deg = np.rad2deg(target_q_rad).tolist()
                        j_str = ", ".join([f"{j:6.2f}" for j in target_joints_deg])
                        print(f"[{self.state}] Z:{self.current_z_target:.3f}m | 虚拟误差:{test_xy_distance*100:.1f}cm | IK:[{j_str}]")
                    else:
                        print(f"[{self.state}] 警告: 当前目标位置不可达 (IK失败)")

            else:
                if self.last_target_id is not None:
                    print(f"--- [状态机] 目标 ID={self.last_target_id} 丢失 ---")
                    self.last_target_id = None
                    self.state = "OBSERVING"
            
            # 5. 频率控制
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.loop_interval - elapsed)
            time.sleep(sleep_time)
