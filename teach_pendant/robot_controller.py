"""
机器人控制器 - 使用单一 WebSocket 连接
"""

import json
import time
import queue
import threading
import numpy as np
from typing import Optional

from .signals import WorkerSignals
from .config import DEFAULT_VEL
from tools.follower_client import FollowerClientWebSocket, FollowerClient


class RobotController:
    """机器人控制器 - 使用单一 WebSocket 连接"""

    def __init__(self, signals: WorkerSignals):
        self.signals = signals
        self.ws_client: Optional[FollowerClientWebSocket] = None
        self.monitor_client: Optional[FollowerClientWebSocket] = None  # 独立监控通道
        self.udp_client: Optional[FollowerClient] = None  # UDP 遥操作客户端
        self.is_running = False
        self.is_logged_in = False
        self.is_enabled = False

        # 跟随模式相关
        self.is_follower_mode = False
        self.jog_coordinate = "joint"  # "joint" 或 "tool"

        # 遥操作相关
        self.udp_connected = False
        self.robot_ip = "192.168.0.10"
        self.udp_port = 9998  # UDP 遥操作端口
        
        self.follower_offset = np.zeros(6)  # 笛卡尔跟随模式下的累积偏移量 [x, y, z, rx, ry, rz] (m, rad)
        self.last_send_time = 0.0  # 上次发送指令的时间
        self.monitor_thread = None
        self.cmd_queue = queue.Queue()

        self.current_joints = [0.0] * 6
        self.actual_tcp = None  # 真实机器人返回的 TCP 位置 [x, y, z, rx, ry, rz] (mm, deg)
        self.data_lock = threading.Lock()

    def connect(self, ip, port):
        """连接 WebSocket (双通道)"""
        try:
            self.robot_ip = ip  # 保存 IP 用于后续 UDP 连接
            
            # 1. 连接控制通道 (通常 5999)
            self.ws_client = FollowerClientWebSocket(ip, port, timeout=5.0)
            if not self.ws_client.connect():
                self.signals.error_occurred.emit(f"控制端口 {port} 连接失败")
                return False

            # 2. 连接监控通道 (固定 5888)
            monitor_port = 5888
            self.monitor_client = FollowerClientWebSocket(ip, monitor_port, timeout=3.0)
            if self.monitor_client.connect():
                print(f"[RobotController] 监控通道连接成功 ({monitor_port})")
            else:
                print(f"[RobotController] 监控通道连接失败 ({monitor_port})，将使用控制通道进行监控(可能会卡顿)")
                self.monitor_client = None

            self.signals.connection_changed.emit(True, "websocket")
            return True
        except Exception as e:
            self.signals.error_occurred.emit(f"连接异常: {e}")
            return False

    def login(self, user="Engineer", password="000000"):
        """登录 (双通道)"""
        try:
            success = False
            # 控制通道登录
            if self.ws_client and self.ws_client.is_connected:
                self.ws_client.logout()
                time.sleep(0.3)
                if self.ws_client.login(user, password):
                    success = True
            
            # 监控通道登录 (如果存在)
            if self.monitor_client and self.monitor_client.is_connected:
                self.monitor_client.logout()
                time.sleep(0.1)
                self.monitor_client.login(user, password) # 监控也需要登录才能读数据

            if success:
                self.is_logged_in = True
                self.signals.status_updated.emit("登录成功")
                return True
            else:
                self.signals.error_occurred.emit("登录失败")
                return False
        except Exception as e:
            self.signals.error_occurred.emit(f"登录失败: {e}")
            return False

    def logout(self):
        """退出登录"""
        try:
            if self.ws_client: self.ws_client.logout()
            if self.monitor_client: self.monitor_client.logout()
            
            self.is_logged_in = False
            self.is_enabled = False
            self.signals.status_updated.emit("已退出登录")
            return True
        except Exception as e:
            self.signals.error_occurred.emit(f"退出登录失败: {e}")
            return False

    def enable(self):
        """使能机器人"""
        try:
            if self.ws_client and self.ws_client.is_connected:
                time.sleep(0.3)
                if self.ws_client.enable_robot():
                    time.sleep(0.3)
                    self.ws_client.set_velocity(100)
                    self.is_enabled = True
                    self.signals.status_updated.emit("使能成功，速度设置为100%")
                    return True
                else:
                    self.signals.error_occurred.emit("使能失败")
                    return False
            else:
                self.signals.error_occurred.emit("WebSocket 未连接")
                return False
        except Exception as e:
            self.signals.error_occurred.emit(f"使能失败: {e}")
            return False

    def start_monitoring(self):
        """启动监控线程 (通过 WebSocket 轮询)"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        """监控循环 - 持续读取关节位置和TCP位置"""
        while self.is_running:
            try:
                # 优先使用独立的监控客户端，避免控制指令阻塞
                client_to_use = self.monitor_client if (self.monitor_client and self.monitor_client.is_connected) else self.ws_client

                if client_to_use and client_to_use.is_connected and self.is_logged_in:
                    # 获取状态
                    client_to_use.send_get_status()

                    # 获取关节位置
                    joint_pos = client_to_use.get_actual_joint_pos()
                    if joint_pos is not None:
                        joints = joint_pos.tolist()
                        with self.data_lock:
                            self.current_joints = joints.copy()
                        self.signals.joints_updated.emit(joints)

                    # 获取 TCP 位置
                    with client_to_use._position_lock:
                        if client_to_use._actual_pe is not None:
                            tcp_data = client_to_use._actual_pe.tolist()
                            with self.data_lock:
                                self.actual_tcp = tcp_data[:6]

                time.sleep(0.05)  # 20Hz
            except Exception as e:
                pass  # 忽略监控异常，继续运行

    def move_joint(self, target_joints, vels=None):
        """发送关节移动指令 (通过 WebSocket)"""
        if not self.is_enabled:
            self.signals.error_occurred.emit("机器人未使能")
            return

        if self.is_follower_mode:
            self.signals.error_occurred.emit("跟随模式下无法使用关节运动，请先停止跟随模式")
            return

        if vels is None:
            vels = DEFAULT_VEL

        try:
            if self.ws_client and self.ws_client.is_connected:
                # 构建 manual_mvaj 命令 (正确格式)
                joint_strs = [f"DOUBLE{{{j:.6f}}}" for j in target_joints]
                joint_inner_str = ",".join(joint_strs)

                vel_strs = [f"DOUBLE{{{v:.6f}}}" for v in vels]
                vel_inner_str = ",".join(vel_strs)

                cmd = f"manual_mvaj --pos=JointTarget{{UrModel_JointTarget{{{joint_inner_str}}}}} --vel=Speed{{{vel_inner_str}}}"
                success, _ = self.ws_client.send_command(cmd, timeout=60)
                if success:
                    self.signals.status_updated.emit(f"移动完成")
                else:
                    self.signals.error_occurred.emit("关节运动指令执行失败")
            else:
                self.signals.error_occurred.emit("WebSocket 未连接")
        except Exception as e:
            self.signals.error_occurred.emit(f"关节运动异常: {e}")

    def jog_joint(self, joint_index, direction, step):
        """Jog单个关节"""
        with self.data_lock:
            target = self.current_joints.copy()
        target[joint_index] += direction * step
        self.move_joint(target)

    def set_velocity(self, percent):
        """设置速度百分比"""
        if self.ws_client and self.ws_client.is_connected:
            self.ws_client.set_velocity(percent)
            self.signals.status_updated.emit(f"速度设置为 {percent}%")

    def stop(self):
        """停止控制器"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        # 停止跟随模式
        self.stop_follower_mode()
        # 关闭 UDP 客户端
        if self.udp_client:
            self.udp_client.close()
            self.udp_client = None
            self.udp_connected = False
        if self.monitor_client:
            self.monitor_client.close()
        if self.ws_client:
            self.ws_client.close()

    # ==========================================
    # 跟随模式独立命令
    # ==========================================

    def cmd_start_follower(self):
        """执行 start_follower 命令"""
        try:
            if not self.ws_client or not self.ws_client.is_connected:
                self.signals.error_occurred.emit("WebSocket 未连接")
                return False

            success, _ = self.ws_client.send_command("start_follower")
            if success:
                self.signals.status_updated.emit("start_follower 已执行")
                return True
            else:
                self.signals.error_occurred.emit("start_follower 失败")
                return False
        except Exception as e:
            self.signals.error_occurred.emit(f"start_follower 异常: {e}")
            return False

    def cmd_set_jog_coordinate_tool(self):
        """执行 set_jog_coordinate --tool 命令 (工具坐标系，用于 Eye-in-Hand 视觉伺服)"""
        try:
            if not self.ws_client or not self.ws_client.is_connected:
                self.signals.error_occurred.emit("WebSocket 未连接")
                return False

            success, _ = self.ws_client.send_command("set_jog_coordinate --tool")
            if success:
                self.jog_coordinate = "tool"
                self.signals.status_updated.emit("set_jog_coordinate --tool 已执行")
                return True
            else:
                self.signals.error_occurred.emit("set_jog_coordinate --tool 失败")
                return False
        except Exception as e:
            self.signals.error_occurred.emit(f"set_jog_coordinate 异常: {e}")
            return False

    def cmd_follower_cart(self):
        """执行 follower_cart 命令 (启动笛卡尔跟随)"""
        try:
            if not self.ws_client or not self.ws_client.is_connected:
                self.signals.error_occurred.emit("WebSocket 未连接")
                return False

            # follower_cart 是持续运行的命令，不等待响应
            self.ws_client._send_command_no_wait("follower_cart")
            self.is_follower_mode = True
            
            # 重置累积偏移量
            self.follower_offset = np.zeros(6)
            print("[RobotController] Follower cart started, offset reset to zero.")

            # 启动状态轮询
            self.ws_client.start_polling(interval=0.05)

            self.signals.status_updated.emit("follower_cart 已启动")
            return True
        except Exception as e:
            self.signals.error_occurred.emit(f"follower_cart 异常: {e}")
            return False

    def cmd_stop_follower(self):
        """执行 stop_follower 命令"""
        try:
            if not self.ws_client or not self.ws_client.is_connected:
                self.signals.error_occurred.emit("WebSocket 未连接")
                return False

            success, _ = self.ws_client.send_command("stop_follower")
            if success:
                self.is_follower_mode = False
                self.signals.status_updated.emit("stop_follower 已执行")
                return True
            else:
                self.signals.error_occurred.emit("stop_follower 失败")
                return False
        except Exception as e:
            self.signals.error_occurred.emit(f"stop_follower 异常: {e}")
            return False

    def start_follower_mode(self, ip):
        """启动跟随模式 (一键执行完整流程: start_follower + set_jog_coordinate --tool + follower_cart)"""
        try:
            if not self.ws_client or not self.ws_client.is_connected:
                self.signals.error_occurred.emit("WebSocket 未连接")
                return False

            self.signals.status_updated.emit("初始化跟随模式...")

            # 使用 init_follower_mode 执行完整的初始化流程
            # login_first=False 因为已经登录
            # skip_enable=False 会在需要时使能
            if self.ws_client.init_follower_mode(login_first=False, skip_enable=self.is_enabled):
                self.is_follower_mode = True
                self.jog_coordinate = "tool"  # 使用工具坐标系 (Eye-in-Hand)
                # 启动状态轮询
                self.ws_client.start_polling(interval=0.05)
                self.signals.status_updated.emit("跟随模式已启动")
                return True
            else:
                self.signals.error_occurred.emit("跟随模式初始化失败")
                return False

        except Exception as e:
            self.signals.error_occurred.emit(f"跟随模式启动失败: {e}")
            return False

    def stop_follower_mode(self):
        """停止跟随模式"""
        if self.ws_client is not None and self.ws_client.is_connected:
            try:
                if self.is_follower_mode:
                    self.ws_client.stop_follower_mode()
            except:
                pass
        self.is_follower_mode = False
        self.signals.status_updated.emit("跟随模式已停止")

    def set_jog_coordinate(self, coordinate):
        """设置 Jog 坐标系 (joint/tool)"""
        try:
            if self.ws_client and self.ws_client.is_connected:
                if coordinate == "tool":
                    success, _ = self.ws_client.send_command("set_jog_coordinate --tool")
                else:
                    success, _ = self.ws_client.send_command("set_jog_coordinate --joint")

                if success:
                    self.jog_coordinate = coordinate
                    self.signals.status_updated.emit(f"坐标系已切换为: {coordinate}")
                    return True
                else:
                    self.signals.error_occurred.emit("坐标系切换失败")
                    return False
            else:
                self.signals.error_occurred.emit("未连接")
                return False
        except Exception as e:
            self.signals.error_occurred.emit(f"设置坐标系失败: {e}")
            return False

    def get_robot_status(self):
        """获取机器人状态信息"""
        status_info = {
            'status': '--',
            'activate': '--',
            'motion': '--',
            'mode': '--',
            'error': '无'
        }

        try:
            if self.ws_client and self.ws_client.is_connected:
                self.ws_client.send_get_status()
                if self.ws_client.last_status:
                    status_info = self._parse_status(self.ws_client.last_status)

            self.signals.robot_status_updated.emit(status_info)
            return status_info

        except Exception as e:
            self.signals.error_occurred.emit(f"获取状态失败: {e}")
            return status_info

    def _parse_status(self, data):
        """解析机器人状态数据"""
        status_info = {
            'status': '--',
            'activate': '--',
            'motion': '--',
            'mode': '--',
            'error': '无'
        }

        try:
            ret_context = data.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)

            robot_msg = ret_context.get('robot_msg', {})

            # 运行状态
            status = robot_msg.get('status', '--')
            status_info['status'] = status

            # 激活状态
            activate = robot_msg.get('activate', '--')
            status_info['activate'] = activate

            # 运动状态
            motion = robot_msg.get('motion', '--')
            status_info['motion'] = motion

            # 模式
            mode = robot_msg.get('mode', '--')
            status_info['mode'] = mode

            # 错误信息
            if status == 'Error':
                error_code = robot_msg.get('error_code', '')
                error_msg = robot_msg.get('error_msg', '未知错误')
                status_info['error'] = f"{error_code}: {error_msg}" if error_code else error_msg
            else:
                status_info['error'] = '无'

        except Exception as e:
            pass

        return status_info

    # ==========================================
    # 遥操作功能 (UDP 发送位姿)
    # ==========================================

    def connect_udp(self, port: int = 9998) -> bool:
        """
        连接 UDP 遥操作端口

        Args:
            port: UDP 端口号 (默认 9998)

        Returns:
            是否连接成功
        """
        try:
            if self.udp_client and self.udp_connected:
                self.signals.status_updated.emit("UDP 已连接")
                return True

            self.udp_port = port
            self.udp_client = FollowerClient(self.robot_ip, port, timeout=1.0)

            if self.udp_client.connect():
                self.udp_connected = True
                self.signals.status_updated.emit(f"UDP 遥操作连接成功 ({self.robot_ip}:{port})")
                return True
            else:
                self.signals.error_occurred.emit(f"UDP 端口 {port} 连接失败")
                return False

        except Exception as e:
            self.signals.error_occurred.emit(f"UDP 连接异常: {e}")
            return False

    def disconnect_udp(self):
        """断开 UDP 连接"""
        if self.udp_client:
            self.udp_client.close()
            self.udp_client = None
        self.udp_connected = False
        self.signals.status_updated.emit("UDP 连接已断开")

    def send_raw_increment(self, dx_mm: float, dy_mm: float, dz_mm: float,
                           drx_deg: float = 0, dry_deg: float = 0, drz_deg: float = 0) -> bool:
        """
        发送笛卡尔增量到机器人 (累积模式)

        解决问题:
        1. 协议是"覆盖"模式 -> 需要在客户端累积增量
        2. 坐标系方向不一致 -> 修正 Y 和 Z 方向
        3. 单次增量过大触发保护 -> 增加安全检查

        Args:
            dx_mm, dy_mm, dz_mm: 位置增量 (毫米) - 用户坐标系
            drx_deg, dry_deg, drz_deg: 角度增量 (度)

        Returns:
            是否发送成功
        """
        if not self.udp_connected or not self.udp_client:
            self.signals.error_occurred.emit("UDP 未连接")
            return False

        if not self.is_follower_mode:
            self.signals.error_occurred.emit("请先启动跟随模式")
            return False

        # 安全检查 1: 单次增量检查 (阈值降低到 5mm)
        limit_mm = 5.0
        if abs(dx_mm) > limit_mm or abs(dy_mm) > limit_mm or abs(dz_mm) > limit_mm:
            print(f"[WARNING] 单次增量超过 {limit_mm}mm，建议减小增量以避免冲击! ({dx_mm}, {dy_mm}, {dz_mm})")
            # 仅警告，不阻止 (允许用户手动 Jog 大步长，但需谨慎)

        # 安全检查 2: 速度检查 (防止高频大增量)
        now = time.time()
        dt = now - self.last_send_time
        
        # 计算本次增量的模长 (米)
        dist_m = np.sqrt((dx_mm/1000.0)**2 + (dy_mm/1000.0)**2 + (dz_mm/1000.0)**2)
        
        if dt > 0.001 and self.last_send_time > 0:  # 避免首次发送或除零
            velocity = dist_m / dt
            max_vel = 0.5  # 最大允许速度 0.5 m/s
            if velocity > max_vel:
                msg = f"指令速度过快 ({velocity:.2f} m/s > {max_vel} m/s)，已拦截! dt={dt*1000:.1f}ms, dist={dist_m*1000:.1f}mm"
                print(f"[ERROR] {msg}")
                self.signals.error_occurred.emit(msg)
                return False

        try:
            # 1. 坐标修正: 仅 Z 轴反向 (根据最终测试结果: X同向, Y同向, Z反向)
            corrected_dx = dx_mm
            corrected_dy = dy_mm
            corrected_dz = -dz_mm  # Z 反向

            # 2. 转换为协议单位 (m, rad)
            dx_m = corrected_dx / 1000.0
            dy_m = corrected_dy / 1000.0
            dz_m = corrected_dz / 1000.0
            drx_rad = np.deg2rad(drx_deg)
            dry_rad = np.deg2rad(dry_deg)
            drz_rad = np.deg2rad(drz_deg)

            # 3. 累积增量
            # self.follower_offset 存储的是: [x, y, z, rx, ry, rz] (m, rad)
            self.follower_offset[0] += dx_m
            self.follower_offset[1] += dy_m
            self.follower_offset[2] += dz_m
            self.follower_offset[3] += drx_rad
            self.follower_offset[4] += dry_rad
            self.follower_offset[5] += drz_rad

            # 4. 打印调试信息
            print(f"[INC] 增量(mm): ({dx_mm}, {dy_mm}, {dz_mm}) -> 修正后(Z反): ({corrected_dx}, {corrected_dy}, {corrected_dz})")
            print(f"[INC] 累积目标(m): {self.follower_offset[:3]}")

            # 5. 发送累积后的总偏移量
            success = self.udp_client.send_pose_euler(
                x=self.follower_offset[0],
                y=self.follower_offset[1],
                z=self.follower_offset[2],
                rx=self.follower_offset[3],
                ry=self.follower_offset[4],
                rz=self.follower_offset[5],
                euler_type="321"
            )

            if success:
                self.last_send_time = now
                self.signals.status_updated.emit(f"已发送累积增量: ({dx_mm}, {dy_mm}, {dz_mm}) mm")

            return success

        except Exception as e:
            self.signals.error_occurred.emit(f"发送失败: {e}")
            return False

    def send_target_pose(self, x: float, y: float, z: float,
                         rz: float, ry: float, rx: float,
                         unit_mm: bool = True, unit_deg: bool = True) -> bool:
        """
        发送目标位姿到机器人 (遥操作)

        注意: 协议要求发送的是增量值，而非绝对位置！
        本方法会自动获取当前位置，计算增量后发送。

        使用 321 (ZYX) 欧拉角顺序:
        - 先绕 Z 轴旋转 rz
        - 再绕 Y 轴旋转 ry
        - 最后绕 X 轴旋转 rx

        Args:
            x, y, z: 目标位置 (绝对位置)
            rz, ry, rx: 目标欧拉角 (321 顺序: Z-Y-X)
            unit_mm: 位置单位是否为毫米 (True=mm, False=m)
            unit_deg: 角度单位是否为度 (True=deg, False=rad)

        Returns:
            是否发送成功
        """
        if not self.udp_connected or not self.udp_client:
            self.signals.error_occurred.emit("UDP 未连接，请先连接遥操作端口")
            return False

        if not self.is_follower_mode:
            self.signals.error_occurred.emit("请先启动跟随模式")
            return False

        try:
            # 获取当前位置 (mm, deg)
            current_tcp = self.get_current_tcp_mm_deg()
            if current_tcp is None:
                self.signals.error_occurred.emit("无法获取当前位置")
                return False

            cur_x, cur_y, cur_z, cur_rx, cur_ry, cur_rz = current_tcp

            # 调试打印：当前位置
            print(f"[DEBUG] 当前位置 (mm, deg): x={cur_x:.2f}, y={cur_y:.2f}, z={cur_z:.2f}, "
                  f"rx={cur_rx:.2f}, ry={cur_ry:.2f}, rz={cur_rz:.2f}")

            # 目标位置单位转换为 mm 和 deg (统一单位后计算增量)
            if unit_mm:
                target_x, target_y, target_z = x, y, z
            else:
                target_x, target_y, target_z = x * 1000.0, y * 1000.0, z * 1000.0

            if unit_deg:
                target_rx, target_ry, target_rz = rx, ry, rz
            else:
                target_rx = np.rad2deg(rx)
                target_ry = np.rad2deg(ry)
                target_rz = np.rad2deg(rz)

            # 调试打印：目标位置
            print(f"[DEBUG] 目标位置 (mm, deg): x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, "
                  f"rx={target_rx:.2f}, ry={target_ry:.2f}, rz={target_rz:.2f}")

            # 计算增量 (mm, deg)
            dx = target_x - cur_x
            dy = target_y - cur_y
            dz = target_z - cur_z
            drx = target_rx - cur_rx
            dry = target_ry - cur_ry
            drz = target_rz - cur_rz

            # 角度归一化到 [-180, 180] 范围，选择最短路径
            def normalize_angle(angle):
                while angle > 180:
                    angle -= 360
                while angle < -180:
                    angle += 360
                return angle

            drx = normalize_angle(drx)
            dry = normalize_angle(dry)
            drz = normalize_angle(drz)

            # 调试打印：增量 (mm, deg) - 原始计算值
            print(f"[DEBUG] 增量 (mm, deg): dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, "
                  f"drx={drx:.2f}, dry={dry:.2f}, drz={drz:.2f}")

            # 坐标修正: Y 和 Z 需要取反 (follower_cart 协议特性)
            corrected_dy = -dy
            corrected_dz = -dz

            print(f"[DEBUG] 修正后 (Y/Z取反): dy={corrected_dy:.2f}, dz={corrected_dz:.2f}")

            # 转换为协议单位 (m, rad)
            dx_m = dx / 1000.0
            dy_m = corrected_dy / 1000.0
            dz_m = corrected_dz / 1000.0
            drx_rad = np.deg2rad(drx)
            dry_rad = np.deg2rad(dry)
            drz_rad = np.deg2rad(drz)

            # 调试打印：发送的数据 (m, rad)
            print(f"[DEBUG] 发送 pe (m, rad): [{dx_m:.6f}, {dy_m:.6f}, {dz_m:.6f}, "
                  f"{drx_rad:.6f}, {dry_rad:.6f}, {drz_rad:.6f}]")

            # 发送增量位姿 (321 欧拉角顺序)
            # pe 数组顺序: [x, y, z, rx, ry, rz]
            # 注意: 直接发送增量，不做坐标变换
            success = self.udp_client.send_pose_euler(
                x=dx_m, y=dy_m, z=dz_m,
                rx=drx_rad, ry=dry_rad, rz=drz_rad,
                euler_type="321"
            )

            if success:
                self.signals.status_updated.emit(
                    f"目标: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) mm, "
                    f"增量: ({dx:.1f}, {dy:.1f}, {dz:.1f}) mm"
                )
            else:
                self.signals.error_occurred.emit("位姿发送失败")

            return success

        except Exception as e:
            self.signals.error_occurred.emit(f"发送位姿异常: {e}")
            return False

    def get_current_tcp_mm_deg(self) -> Optional[tuple]:
        """
        获取当前 TCP 位置 (毫米, 度)

        Returns:
            (x, y, z, rx, ry, rz) 或 None
            位置单位: mm, 角度单位: deg
        """
        with self.data_lock:
            if self.actual_tcp is not None:
                # actual_tcp 已经是 [x, y, z, rx, ry, rz] 格式 (mm, deg)
                return tuple(self.actual_tcp)
        return None
