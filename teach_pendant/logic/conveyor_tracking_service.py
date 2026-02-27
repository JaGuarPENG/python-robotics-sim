"""
传送带动态追踪与触碰服务 (阶段二: 基础 30Hz 追踪控制流)
负责：以 30Hz 频率获取绿色小球坐标，计算悬停 20cm 处的 IK 逆解，并打印验证。
"""

import threading
import time
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class ConveyorTrackingService(QObject):
    # 定义信号，可用于后续将状态、错误信息抛回主界面
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
        
        # 为了避免连续打印刷屏，保存上一次的ID
        self.last_target_id = None

    def start_tracking(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        self.status_updated.emit("传送带追踪服务已启动")
        print("[阶段二] 传送带追踪服务已启动，等待目标进入视野...")

    def stop_tracking(self):
        if not self.is_running:
            return
            
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        self.status_updated.emit("传送带追踪服务已停止")
        print("[阶段二] 传送带追踪服务已停止")
        self.last_target_id = None

    def _tracking_loop(self):
        while self.is_running:
            loop_start_time = time.perf_counter()
            
            # 1. 尝试获取当前追踪目标
            target_pos, target_id = None, None
            if hasattr(self.robot_view, 'renderer') and self.robot_view.renderer:
                target_pos, target_id = self.robot_view.renderer.get_tracking_target()
                
            if target_pos is not None and target_id is not None:
                # 2. 计算悬停目标位姿: XY 追踪小球，Z 轴抬高 20cm
                hover_z = target_pos[2] + 0.20
                hover_pos = [target_pos[0], target_pos[1], hover_z]
                
                # 设定末端姿态: 末端朝下 (rx=180, ry=0, rz=0 是一个比较标准的向下姿态，具体取决于您的DH系)
                # 这里我们假设使用默认向下的姿态
                hover_rpy_deg = [180.0, 0.0, 0.0] 
                
                # 3. 调用 FastIK 进行逆解计算
                target_q_rad, ik_time = self.fast_ik.solve_ik(
                    pos=hover_pos,
                    rpy_deg=hover_rpy_deg,
                    current_joints=np.deg2rad(self.controller.state.get_joints())
                )
                
                # 4. [阶段二验证] 打印输出
                if target_q_rad is not None:
                    target_joints_deg = np.rad2deg(target_q_rad).tolist()
                    if self.last_target_id != target_id:
                        print(f"\n--- 发现新目标 ID={target_id} ---")
                        self.last_target_id = target_id
                    
                    # 仅在后台验证，暂时不下发给机械臂
                    # 格式化打印以便于观察
                    j_str = ", ".join([f"{j:6.2f}" for j in target_joints_deg])
                    print(f"[阶段二IK] 目标ID:{target_id} | 期望位置:[{hover_pos[0]:.3f}, {hover_pos[1]:.3f}, {hover_pos[2]:.3f}] | IK解算关节角:[{j_str}]")
                else:
                    print(f"[阶段二IK] 警告: 目标ID={target_id} 的悬停位置不可达 (IK解算失败)")

            else:
                if self.last_target_id is not None:
                    print(f"--- 目标 ID={self.last_target_id} 丢失 ---")
                    self.last_target_id = None
            
            # 5. 频率控制
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.loop_interval - elapsed)
            time.sleep(sleep_time)
