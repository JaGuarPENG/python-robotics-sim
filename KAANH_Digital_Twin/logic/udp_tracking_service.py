"""
UDP 遥操作追踪服务 (绝对偏移模式 - 与轨迹执行逻辑对齐)
负责：下发相对于开启跟随模式时起点的[总偏移量]，确保运动速度与传送带完全同步。
"""

import threading
import time
import math
import numpy as np
import sys
import os
from PyQt5.QtCore import QObject, pyqtSignal

# 添加 tools 目录到路径以导入 FollowerClient
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
try:
    from follower_client import FollowerClient
except ImportError:
    FollowerClient = None

class UDPTrackingService(QObject):
    status_updated = pyqtSignal(str)
    
    def __init__(self, controller, robot_view):
        super().__init__()
        self.controller = controller
        self.robot_view = robot_view
        
        self.is_running = False
        self.tracking_thread = None
        self.loop_hz = 30.0
        self.loop_interval = 1.0 / self.loop_hz
        
        # === 状态机变量 ===
        self.last_target_id = None
        self.state = "OBSERVING"
        self.hover_count = 0
        self.origin_tcp_m = None  # 唯一的 UDP 坐标系原点
        
        # === 遥操作 UDP 客户端 ===
        self.udp_client = None
        
        # === 运动控制参数 (米) ===
        self.hover_height = 0.10
        self.target_z_surface = 0.211
        self.current_z_target = self.target_z_surface + self.hover_height
        
        self.approach_speed_z = 0.05
        self.xy_threshold = 0.010
        
        # === 动态补偿参数 ===
        self.look_ahead_frames = 1.2    # 略微增加预瞄补偿
        self.conveyor_speed_y = 0.05
        self.visual_offset_z = 0.050

    def start_tracking(self, ip="192.168.0.10"):
        if self.is_running: return
        self.is_running = True
        self.state = "INITIALIZING"
        self.tracking_thread = threading.Thread(target=self._tracking_worker, args=(ip,), daemon=True)
        self.tracking_thread.start()

    def stop_tracking(self):
        if not self.is_running: return
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        self.controller.cmd_stop_follower()
        if self.udp_client:
            self.udp_client.close()
        self.status_updated.emit("UDP 追踪服务已停止")

    def _tracking_worker(self, ip):
        """完全在子线程运行初始化，防止 UI 卡死"""
        try:
            self.udp_client = FollowerClient(ip, 9998)
            if not self.udp_client.connect():
                self.status_updated.emit("UDP 连接失败")
                self.is_running = False
                return

            # 1. 自动归位
            self.status_updated.emit("正在自动归位机器人...")
            home_joints = [0, -15, 105, 0, -90, 0]
            self.controller.move_joint(home_joints, vels=[60]*6, wait_for_finish=True)
            time.sleep(0.5)

            # 2. 开启跟随模式 (这一瞬间确定了 UDP 的 [0,0,0])
            self.status_updated.emit("正在开启工控机跟随模式...")
            if not self.controller.start_follower_mode(ip):
                self.status_updated.emit("跟随模式启动失败")
                self.is_running = False
                return
            
            # 必须等待一下让模式切换彻底完成
            time.sleep(0.8)
            
            # 3. 核心：捕获唯一的基准原点 (Origin)
            # 遥操作模式下，控制器将此时的位置视为增量的起点
            init_tcp = self.controller.state.get_tcp()
            self.origin_tcp_m = np.array([init_tcp[0]/1000.0, init_tcp[1]/1000.0, init_tcp[2]/1000.0])
            print(f"[UDP模式] 基准点已锁定: {self.origin_tcp_m}")

            self.state = "OBSERVING"
            self.status_updated.emit("UDP 绝对增量模式已就绪")
            
            # 进入主循环
            self._tracking_loop()

        except Exception as e:
            self.status_updated.emit(f"追踪异常: {e}")
            self.is_running = False

    def _tracking_loop(self):
        while self.is_running:
            loop_start_time = time.perf_counter()
            
            target_pos, target_id = None, None
            if hasattr(self.robot_view, 'renderer') and self.robot_view.renderer:
                target_pos, target_id = self.robot_view.renderer.get_tracking_target()
                
            if target_pos is not None and target_id is not None:
                # 目标切换检测
                if self.last_target_id != target_id:
                    self.last_target_id = target_id
                    self.state = "HOVERING"
                    self.hover_count = 0
                    self.current_z_target = target_pos[2] + self.hover_height
                
                # 获取实时反馈 (仅用于状态判定和日志)
                tcp_mm = self.controller.state.get_tcp()
                tcp_m = np.array([tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0])
                
                # 预测目标绝对位置 (Y轴预瞄)
                dt_comp = self.loop_interval * self.look_ahead_frames
                predicted_y = target_pos[1] + (self.conveyor_speed_y * dt_comp)
                
                # 计算物理目标 Z (视觉 + 偏置 + 探针)
                phys_target_z = self.current_z_target + self.visual_offset_z + 0.20

                # ==========================
                # 计算相对于 Origin 的总偏移 (Total Delta)
                # ==========================
                # 这正是轨迹执行模式的做法
                total_dx = target_pos[0] - self.origin_tcp_m[0]
                total_dy = predicted_y - self.origin_tcp_m[1]
                total_dz = phys_target_z - self.origin_tcp_m[2]

                # 实时误差 (判定是否对准)
                real_xy_dist = math.sqrt((target_pos[0]-tcp_m[0])**2 + (target_pos[1]-tcp_m[1])**2)

                # 状态机逻辑
                if self.state == "HOVERING":
                    # 稳定判定
                    if real_xy_dist < self.xy_threshold and abs(tcp_m[2] - phys_target_z) < 0.050:
                        self.hover_count += 1
                    else:
                        self.hover_count = 0
                        
                    if self.hover_count >= 5:
                        print(f"--- [UDP状态机] 稳定对准，开始下压 ---")
                        self.state = "APPROACHING"
                        self.hover_count = 0

                elif self.state == "APPROACHING":
                    self.current_z_target -= self.approach_speed_z * self.loop_interval
                    if self.current_z_target < self.target_z_surface - 0.010:
                        self.current_z_target = self.target_z_surface - 0.010
                    
                    # 触碰判定 (物理高度降到表面)
                    if tcp_m[2] <= 0.465: 
                        print(f"--- [UDP状态机] 已触碰，误差:{real_xy_dist*1000:.1f}mm ---")
                        self.state = "RETURNING"
                        self.robot_view.renderer.mark_target_reached()

                elif self.state == "RETURNING":
                    self.current_z_target += (self.approach_speed_z * 3) * self.loop_interval
                    if self.current_z_target >= target_pos[2] + self.hover_height:
                        self.state = "OBSERVING"
                        self.last_target_id = None

                # ==========================
                # 执行运动 (轴映射下发)
                # 严格参考轨迹执行逻辑: send_pose_euler(total_dy, total_dx, -total_dz)
                # ==========================
                if self.state != "OBSERVING":
                    # 下发总偏移量，工控机会根据这个值实时计算速度
                    self.udp_client.send_pose_euler(total_dy, total_dx, -total_dz, 0, 0, 0)
                    
                    if int(loop_start_time * 10) % 10 == 0:
                        print(f"[UDP-{self.state}] 指令偏移:[{total_dy*1000:.1f}, {total_dx*1000:.1f}, {-total_dz*1000:.1f}]mm | XY误差:{real_xy_dist*1000:.1f}mm")

            else:
                if self.state != "OBSERVING" and self.state != "INITIALIZING":
                    self.state = "OBSERVING"
                    self.last_target_id = None
                
            elapsed = time.perf_counter() - loop_start_time
            time.sleep(max(0, self.loop_interval - elapsed))
