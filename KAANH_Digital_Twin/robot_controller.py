"""
机器人控制器 (重构后) - 通信分发门面
"""

import json
import time
import threading
import numpy as np
from typing import Optional

from .signals import WorkerSignals
from .config import DEFAULT_VEL
from .core.robot_state import RobotState
from .core.safety_guard import SafetyGuard
from tools.follower_client import FollowerClientWebSocket, FollowerClient


class RobotController:
    """机器人控制器 - 负责底层通信与指令分发"""

    def __init__(self, signals: WorkerSignals):
        self.signals = signals
        self.state = RobotState()
        self.safety = SafetyGuard()
        
        self.ws_client: Optional[FollowerClientWebSocket] = None
        self.monitor_client: Optional[FollowerClientWebSocket] = None
        self.udp_client: Optional[FollowerClient] = None
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.robot_ip = "192.168.1.10"
        
        # 缓存累积偏移量
        self.follower_offset = np.zeros(6)

    # --- 连接管理 ---

    def connect(self, ip, port):
        try:
            self.robot_ip = ip
            self.ws_client = FollowerClientWebSocket(ip, port, timeout=5.0)
            if not self.ws_client.connect():
                self.signals.error_occurred.emit(f"控制端口 {port} 连接失败")
                return False

            self.monitor_client = FollowerClientWebSocket(ip, 5888, timeout=3.0)
            if not self.monitor_client.connect():
                print("[RobotController] 监控通道连接失败，回退到主通道")
                self.monitor_client = None

            self.state.is_connected = True
            self.signals.connection_changed.emit(True, "websocket")
            return True
        except Exception as e:
            self.signals.error_occurred.emit(f"连接异常: {e}")
            return False

    def login(self, user="Manufacturer", password="2045"):
        success = False
        if self.ws_client and self.ws_client.is_connected:
            self.ws_client.logout()
            time.sleep(0.3)
            if self.ws_client.login(user, password):
                success = True
        
        if success:
            if self.monitor_client:
                self.monitor_client.login(user, password)
            self.state.is_logged_in = True
            self.signals.status_updated.emit("登录成功")
        return success

    def logout(self):
        if self.ws_client: self.ws_client.logout()
        self.state.is_logged_in = False
        self.state.is_enabled = False
        self.signals.status_updated.emit("已退出登录")
        return True

    def enable(self):
        if not self.ws_client: return False
        time.sleep(0.3)
        if self.ws_client.enable_robot():
            time.sleep(0.3)
            self.ws_client.set_velocity(100)
            self.state.is_enabled = True
            self.signals.status_updated.emit("使能成功，速度设置为100%")
            return True
        return False

    # --- 监控循环 ---

    def start_monitoring(self):
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        status_counter = 0
        reconnect_counter = 0
        while self.is_monitoring:
            try:
                # 自动重连逻辑
                if self.ws_client and not self.ws_client.is_connected:
                    reconnect_counter += 1
                    if reconnect_counter >= 20: # 约每秒尝试一次
                        reconnect_counter = 0
                        print("[RobotController] 检测到连接断开，尝试自动重连...")
                        if self.ws_client.connect():
                            if self.state.is_logged_in:
                                self.login() # 重新登录
                                if self.state.is_enabled:
                                    self.enable() # 重新使能
                
                client = self.monitor_client if self.monitor_client else self.ws_client
                if client and client.is_connected and self.state.is_logged_in:
                    client.send_get_status()
                    
                    # 更新关节
                    joints = client.get_actual_joint_pos()
                    if joints is not None:
                        self.state.update_joints(joints.tolist())
                        self.signals.joints_updated.emit(joints.tolist())

                    # 更新 TCP
                    with client._position_lock:
                        if client._actual_pe is not None:
                            self.state.update_tcp(client._actual_pe.tolist()[:6])

                    # 每 0.5s 自动刷新一次详细状态
                    status_counter += 1
                    if status_counter >= 10:
                        status_counter = 0
                        if client.last_status:
                            info = self._parse_status(client.last_status)
                            self.state.update_status_info(info)
                            self.signals.robot_status_updated.emit(info)
                
                time.sleep(0.05)
            except Exception as e:
                print(f"[RobotController] 监控循环异常: {e}")
                time.sleep(0.5)

    # --- 核心控制方法 ---

    def move_joint(self, target_joints, vels=None, wait_for_finish=True):
        if not self.state.is_enabled:
            self.signals.error_occurred.emit("机器人未使能")
            return

        vels = vels or DEFAULT_VEL
        try:
            # 构建指令
            joint_inner = ",".join([f"DOUBLE{{{j:.6f}}}" for j in target_joints])
            vel_inner = ",".join([f"DOUBLE{{{v:.6f}}}" for v in vels])
            cmd = f"manual_mvaj --pos=JointTarget{{UrModel_JointTarget{{{joint_inner}}}}} --vel=Speed{{{vel_inner}}}"
            
            if wait_for_finish:
                success, _ = self.ws_client.send_command(cmd, timeout=60)
                if success:
                    # 循环检查直到运动停止 (Motion 变为 Stop 或 Idle)
                    time.sleep(0.5) # 先等指令生效
                    for _ in range(200): # 最多等待 10s
                        if self.state.status_info.get('motion') in ['Stop', 'Idle', '--']:
                            break
                        time.sleep(0.05)
                    self.signals.status_updated.emit("移动完成")
            else:
                # 异步模式：发了就走
                self.ws_client._send_command_no_wait(cmd)
                
        except Exception as e:
            self.signals.error_occurred.emit(f"运动异常: {e}")

    def jog_joint(self, joint_index, direction, step):
        target = self.state.get_joints()
        target[joint_index] += direction * step
        self.move_joint(target, wait_for_finish=False)

    def set_velocity(self, percent):
        if self.ws_client:
            self.ws_client.set_velocity(percent)
            self.signals.status_updated.emit(f"速度设置为 {percent}%")

    # --- 跟随模式命令 (Facade) ---

    def cmd_start_follower(self):
        if self.ws_client.send_command("start_follower")[0]:
            self.signals.status_updated.emit("start_follower 已执行")
            return True
        return False

    def cmd_set_jog_coordinate_tool(self):
        if self.ws_client.send_command("set_jog_coordinate --tool")[0]:
            self.signals.status_updated.emit("坐标系已切换为工具坐标系")
            return True
        return False

    def cmd_follower_cart(self):
        self.ws_client._send_command_no_wait("follower_cart")
        self.state.is_follower_mode = True
        self.follower_offset.fill(0)
        self.ws_client.start_polling(interval=0.05)
        self.signals.status_updated.emit("follower_cart 已启动")
        return True

    def cmd_stop_follower(self):
        if self.ws_client.send_command("stop_follower")[0]:
            self.state.is_follower_mode = False
            self.signals.status_updated.emit("跟随模式已停止")
            return True
        return False

    def init_udp_tracking_mode(self):
        """初始化 UDP follower_cart 追踪模式，返回零点 P0（米，基座坐标系）"""
        self.cmd_start_follower()
        time.sleep(1.0)
        self.cmd_set_jog_coordinate_tool()
        time.sleep(0.5)
        self.cmd_follower_cart()   # 内部会重置 follower_offset
        time.sleep(1.0)
        tcp_mm = self.state.get_tcp()
        return [tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0]

    def send_udp_target(self, target_m, p0_m):
        """
        发送 UDP 偏移目标（基座坐标系 -> 工具坐标系映射）
        使用与单点遥操作一致的坐标映射：
        - X 直接对应
        - Y 需要取反
        - Z 需要取反
        """
        # 计算基座坐标系下的偏移量
        offset_x = target_m[0] - p0_m[0]  # 基座X偏移
        offset_y = target_m[1] - p0_m[1]  # 基座Y偏移
        offset_z = target_m[2] - p0_m[2]  # 基座Z偏移
        
        # 坐标映射：基座 → 工具坐标系
        send_x = offset_x      # X 直接对应
        send_y = -offset_y     # Y 取反
        send_z = -offset_z     # Z 取反
        
        self.udp_client.send_pose_euler(send_x, send_y, send_z, 0, 0, 0)

    def start_follower_mode(self, ip):
        if self.ws_client.init_follower_mode(login_first=False, skip_enable=self.state.is_enabled):
            self.state.is_follower_mode = True
            self.ws_client.start_polling(interval=0.05)
            return True
        return False

    # --- UDP 遥操作相关 ---

    def connect_udp(self, port=9998):
        self.udp_client = FollowerClient(self.robot_ip, port)
        if self.udp_client.connect():
            self.signals.status_updated.emit(f"UDP 连接成功: {port}")
            return True
        return False

    def disconnect_udp(self):
        if self.udp_client: self.udp_client.close()
        self.signals.status_updated.emit("UDP 连接已断开")

    def send_raw_increment(self, dx, dy, dz, drx=0, dry=0, drz=0):
        if not self.udp_client: return False
        
        # 安全检查
        ok, msg = self.safety.check_increment(dx, dy, dz)
        if not ok: 
            self.signals.error_occurred.emit(msg)
            return False
            
        ok, msg = self.safety.check_velocity(dx, dy, dz)
        if not ok:
            self.signals.error_occurred.emit(msg)
            return False

        # 逻辑处理：修正 Z 轴方向，累积偏移
        self.follower_offset[0] += dx / 1000.0
        self.follower_offset[1] += dy / 1000.0
        self.follower_offset[2] += -dz / 1000.0 # 修正
        self.follower_offset[3:] += np.deg2rad([drx, dry, drz])

        success = self.udp_client.send_pose_euler(*self.follower_offset, euler_type="321")
        return success

    def get_robot_status(self):
        if self.ws_client:
            self.ws_client.send_get_status()
            if self.ws_client.last_status:
                info = self._parse_status(self.ws_client.last_status)
                self.state.update_status_info(info)
                self.signals.robot_status_updated.emit(info)
                return info
        return {}

    def _parse_status(self, data):
        # 逻辑保持不变，仅用于解析
        try:
            ctx = data.get('ret_context', {})
            if isinstance(ctx, str): ctx = json.loads(ctx)
            msg = ctx.get('robot_msg', {})
            return {
                'status': msg.get('status', '--'),
                'activate': msg.get('activate', '--'),
                'motion': msg.get('motion', '--'),
                'mode': msg.get('mode', '--'),
                'error': f"{msg.get('error_code')}: {msg.get('error_msg')}" if msg.get('status')=='Error' else '无'
            }
        except: return {}

    def get_current_tcp_mm_deg(self):
        return self.state.get_tcp()

    def stop(self):
        self.is_monitoring = False
        self.cmd_stop_follower()
        self.disconnect_udp()
        if self.ws_client: self.ws_client.close()
        if self.monitor_client: self.monitor_client.close()

    @property
    def current_joints(self):
        return self.state.get_joints()

    @property
    def actual_tcp(self):
        return self.state.get_tcp()

    @property
    def is_enabled(self):
        return self.state.is_enabled

    @property
    def is_follower_mode(self):
        return self.state.is_follower_mode

    @property
    def udp_connected(self):
        return self.udp_client.is_connected if self.udp_client else False