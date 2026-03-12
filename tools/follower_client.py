"""
遥操作客户端 - 基于 UDP 协议发送位姿指令到机器人控制器
参考 follower_demo C++ 实现
"""

import socket
import json
import struct
import time
import threading
import hashlib
import numpy as np
from typing import List, Optional, Tuple


class FollowerClient:
    """
    遥操作跟随客户端

    通过 UDP 协议向机器人控制器发送笛卡尔位姿指令，实现视觉随动伺服。

    协议格式 (JSON):
    {
        "type": "321",  # 欧拉角类型 (ZYX)
        "pe": [[x, y, z, rx, ry, rz]],  # 位置 + 欧拉角 (米, 弧度)
        "pq": []  # 位置 + 四元数 (可选)
    }
    """

    # ARIS 消息头格式
    HEADER_FORMAT = '<IIQqqq'  # msg_len, msg_id, msg_type, reserved1, reserved2, reserved3
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(self, ip: str = "192.168.0.10", port: int = 9998, timeout: float = 1.0):
        """
        初始化遥操作客户端

        Args:
            ip: 机器人控制器 IP 地址
            port: UDP 端口号 (默认 9998)
            timeout: 超时时间 (秒)
        """
        self.ip = ip
        self.port = port
        self.timeout = timeout

        self.socket: Optional[socket.socket] = None
        self.is_connected = False

        # 接收线程
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_response: Optional[bytes] = None
        self._response_lock = threading.Lock()

        # 机器人实际位置 (从反馈中解析)
        self._actual_pe: Optional[np.ndarray] = None  # [x, y, z, rx, ry, rz]
        self._actual_pq: Optional[np.ndarray] = None  # [x, y, z, qx, qy, qz, qw]
        self._actual_joint_pos: Optional[np.ndarray] = None  # [j1, j2, j3, j4, j5, j6]
        self._position_lock = threading.Lock()

    def connect(self) -> bool:
        """建立 UDP 连接"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)

            # UDP 不需要真正的连接，但可以用 connect() 绑定目标地址
            self.socket.connect((self.ip, self.port))

            self.is_connected = True
            self._running = True

            # 启动接收线程
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

            print(f"[FollowerClient] 已连接到 {self.ip}:{self.port}")
            return True

        except Exception as e:
            print(f"[FollowerClient] 连接失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
        self.is_connected = False
        print("[FollowerClient] 连接已关闭")

    def _pack_message(self, payload: bytes) -> bytes:
        """
        打包 ARIS 消息格式

        Args:
            payload: JSON 字符串的字节数据

        Returns:
            完整的消息包 (头部 + 数据)
        """
        msg_len = len(payload)
        msg_id = 0
        msg_type = 0
        reserved1 = 0
        reserved2 = 0
        reserved3 = 0

        header = struct.pack(
            self.HEADER_FORMAT,
            msg_len, msg_id, msg_type, reserved1, reserved2, reserved3
        )
        return header + payload

    def _recv_loop(self):
        """接收响应的后台线程"""
        while self._running:
            try:
                data, addr = self.socket.recvfrom(4096)
                if len(data) > self.HEADER_SIZE:
                    response_data = data[self.HEADER_SIZE:]
                    with self._response_lock:
                        self._last_response = response_data
                    # 解析机器人状态
                    self._parse_robot_state(response_data)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[FollowerClient] 接收错误: {e}")
                break

    def _parse_robot_state(self, data: bytes):
        """
        解析机器人返回的状态数据，提取实际位置

        返回数据格式 (JSON):
        {
            "ret_code": 0,
            "ret_context": {
                "motion_msg": {
                    "pe": [[x, y, z, rx, ry, rz]],
                    "pq": [[x, y, z, qx, qy, qz, qw]],
                    "actual_pos": [[j1, j2, j3, j4, j5, j6]]
                }
            }
        }
        """
        try:
            response = json.loads(data.decode('utf-8'))

            # 检查返回码
            if response.get('ret_code', -1) != 0:
                return

            # 解析 ret_context (可能是字符串或字典)
            ret_context = response.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)

            # 提取 motion_msg
            motion_msg = ret_context.get('motion_msg', {})

            with self._position_lock:
                # 提取末端位姿 (欧拉角)
                pe = motion_msg.get('pe', [])
                if pe and len(pe) > 0 and len(pe[0]) >= 6:
                    # 工控机返回的旋转顺序是 [rz, ry, rx]，需要转换为 [rx, ry, rz]
                    # pe[0][3]=rz, pe[0][4]=ry, pe[0][5]=rx
                    pe_raw = pe[0][:6]
                    self._actual_pe = np.array([
                        pe_raw[0], pe_raw[1], pe_raw[2],  # x, y, z
                        pe_raw[5], pe_raw[4], pe_raw[3]   # rx, ry, rz (交换顺序)
                    ], dtype=np.float64)

                # 提取末端位姿 (四元数)
                pq = motion_msg.get('pq', [])
                if pq and len(pq) > 0 and len(pq[0]) >= 7:
                    self._actual_pq = np.array(pq[0][:7], dtype=np.float64)

                # 提取关节角度
                actual_pos = motion_msg.get('actual_pos', [])
                if actual_pos and len(actual_pos) > 0 and len(actual_pos[0]) >= 6:
                    self._actual_joint_pos = np.array(actual_pos[0][:6], dtype=np.float64)

        except json.JSONDecodeError:
            pass  # 非 JSON 格式的响应，忽略
        except Exception as e:
            pass  # 解析错误，忽略

    def send_get_status(self) -> bool:
        """
        发送 get 指令获取机器人状态

        Returns:
            是否发送成功
        """
        if not self.is_connected:
            return False

        try:
            payload = b"get"
            packet = self._pack_message(payload)
            self.socket.send(packet)
            return True
        except Exception as e:
            print(f"[FollowerClient] 发送 get 失败: {e}")
            return False

    def get_actual_position(self) -> Optional[np.ndarray]:
        """
        获取机器人末端实际位置 (欧拉角表示)

        Returns:
            [x, y, z, rx, ry, rz] 或 None (单位: 米, 弧度)
            注意: 返回的位置单位可能是毫米和度，需要根据实际情况转换
        """
        with self._position_lock:
            if self._actual_pe is not None:
                return self._actual_pe.copy()
        return None

    def get_actual_position_mm(self) -> Optional[np.ndarray]:
        """
        获取机器人末端实际位置，转换为米

        Returns:
            [x, y, z] (单位: 米) 或 None
        """
        with self._position_lock:
            if self._actual_pe is not None:
                # pe 返回的位置单位是毫米，角度是度
                # 转换为米
                pos = self._actual_pe[:3].copy() / 1000.0
                return pos
        return None

    def get_actual_pose_m(self) -> Optional[np.ndarray]:
        """
        获取机器人末端实际位姿，转换为米和弧度

        Returns:
            [x, y, z, rx, ry, rz] (单位: 米, 弧度) 或 None
        """
        with self._position_lock:
            if self._actual_pe is not None:
                pose = self._actual_pe.copy()
                # 位置: 毫米 -> 米
                pose[:3] = pose[:3] / 1000.0
                # 角度: 度 -> 弧度
                pose[3:] = np.deg2rad(pose[3:])
                return pose
        return None

    def get_actual_joint_pos(self) -> Optional[np.ndarray]:
        """
        获取机器人关节实际角度

        Returns:
            [j1, j2, j3, j4, j5, j6] (单位: 度) 或 None
        """
        with self._position_lock:
            if self._actual_joint_pos is not None:
                return self._actual_joint_pos.copy()
        return None

    def send_pose_euler(self, x: float, y: float, z: float,
                        rx: float, ry: float, rz: float,
                        euler_type: str = "321") -> bool:
        """
        发送笛卡尔位姿 (欧拉角表示)

        Args:
            x, y, z: 位置 (米)
            rx, ry, rz: 欧拉角 (弧度)
            euler_type: 欧拉角类型 ("321" = ZYX, "123" = XYZ 等)

        Returns:
            是否发送成功
        """
        if not self.is_connected:
            print("[FollowerClient] 未连接")
            return False

        try:
            # 构建 JSON 消息
            msg = {
                "type": euler_type,
                "pe": [[x, y, z, rx, ry, rz]],
                "pq": []
            }

            payload = json.dumps(msg).encode('utf-8')
            packet = self._pack_message(payload)

            self.socket.send(packet)
            return True

        except Exception as e:
            print(f"[FollowerClient] 发送失败: {e}")
            return False

    def send_pose_quaternion(self, x: float, y: float, z: float,
                             qx: float, qy: float, qz: float, qw: float) -> bool:
        """
        发送笛卡尔位姿 (四元数表示)

        Args:
            x, y, z: 位置 (米)
            qx, qy, qz, qw: 四元数

        Returns:
            是否发送成功
        """
        if not self.is_connected:
            print("[FollowerClient] 未连接")
            return False

        try:
            msg = {
                "type": "321",
                "pe": [],
                "pq": [[x, y, z, qx, qy, qz, qw]]
            }

            payload = json.dumps(msg).encode('utf-8')
            packet = self._pack_message(payload)

            self.socket.send(packet)
            return True

        except Exception as e:
            print(f"[FollowerClient] 发送失败: {e}")
            return False

    def send_delta_pose(self, dx: float, dy: float, dz: float,
                        drx: float = 0.0, dry: float = 0.0, drz: float = 0.0) -> bool:
        """
        发送增量位姿 (相对于当前位置的偏移)

        Args:
            dx, dy, dz: 位置增量 (米)
            drx, dry, drz: 姿态增量 (弧度)

        Returns:
            是否发送成功
        """
        return self.send_pose_euler(dx, dy, dz, drx, dry, drz)

    def get_last_response(self) -> Optional[dict]:
        """获取最后一次响应"""
        with self._response_lock:
            if self._last_response:
                try:
                    return json.loads(self._last_response.decode('utf-8'))
                except:
                    return None
        return None


class FollowerClientWebSocket:
    """
    基于 WebSocket 的状态查询客户端

    通过 WebSocket 端口 5999 发送 get 指令获取机器人状态
    """

    def __init__(self, ip: str = "192.168.0.10", port: int = 5999, timeout: float = 5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.ws = None
        self.is_connected = False

        # 机器人实际位置
        self._actual_pe: Optional[np.ndarray] = None
        self._actual_pq: Optional[np.ndarray] = None
        self._actual_joint_pos: Optional[np.ndarray] = None
        self._position_lock = threading.Lock()
        
        # 新增：WebSocket 通信锁，防止多线程竞争
        self._sock_lock = threading.Lock()

        # 最后一次状态查询结果
        self.last_status: Optional[dict] = None

        # 后台轮询线程
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """建立 WebSocket 连接"""
        with self._sock_lock:
            try:
                import websocket
                if self.ws:
                    try: self.ws.close()
                    except: pass
                
                self.ws = websocket.WebSocket()
                self.ws.settimeout(self.timeout)
                self.ws.connect(f"ws://{self.ip}:{self.port}")
                self.is_connected = True
                print(f"[WebSocket] 已连接到 ws://{self.ip}:{self.port}")
                return True
            except ImportError:
                print("[WebSocket] 需要安装 websocket-client: pip install websocket-client")
                return False
            except Exception as e:
                self.is_connected = False
                print(f"[WebSocket] 连接失败: {e}")
                return False

    def close(self):
        """关闭连接"""
        self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1)
        
        with self._sock_lock:
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
            self.ws = None
            self.is_connected = False
        print("[WebSocket] 连接已关闭")

    def _pack_header(self, msg_len: int) -> bytes:
        """打包消息头"""
        return struct.pack('<IIQqqq', msg_len, 0, 0, 0, 0, 0)

    def send_command(self, command: str, timeout: float = None) -> Tuple[bool, Optional[dict]]:
        """
        发送控制指令到机器人

        Args:
            command: 指令字符串，如 "start_follower", "follower_cart" 等
            timeout: 超时时间 (秒)，None 表示使用默认超时

        Returns:
            (是否成功, 响应数据)
        """
        if not self.is_connected or not self.ws:
            # print(f"[WebSocket] 未连接，无法发送指令: {command}")
            return False, None

        with self._sock_lock:
            try:
                import websocket

                # 临时设置超时时间
                if timeout is not None:
                    self.ws.settimeout(timeout)

                payload = command.encode('utf-8')
                packet = self._pack_header(len(payload)) + payload
                self.ws.send(packet, opcode=websocket.ABNF.OPCODE_BINARY)

                # 接收响应
                resp = self.ws.recv()

                # 恢复默认超时
                if timeout is not None:
                    self.ws.settimeout(self.timeout)

                if len(resp) > 40:
                    data = resp[40:]
                    try:
                        response = json.loads(data.decode('utf-8'))
                        ret_code = response.get('ret_code', -1)
                        ret_msg = response.get('ret_msg', '')
                        if ret_code != 0:
                            print(f"[WebSocket] 指令 '{command}' 返回: code={ret_code}, msg={ret_msg}")
                        return ret_code == 0, response
                    except json.JSONDecodeError:
                        print(f"[WebSocket] 指令 '{command}' 响应解析失败")
                        return False, None
                return False, None

            except Exception as e:
                print(f"[WebSocket] 发送指令 '{command}' 失败: {e}")
                self.is_connected = False # 标记连接已断开
                return False, None

    def check_robot_enabled(self) -> bool:
        """
        检查机器人是否已使能

        Returns:
            是否已使能
        """
        if self.last_status is None:
            self.send_get_status()

        if self.last_status is None:
            return False

        try:
            ret_context = self.last_status.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)

            robot_msg = ret_context.get('robot_msg', {})
            activate = robot_msg.get('activate', 'Disabled')

            # 检查是否已激活/使能
            is_enabled = activate.lower() in ['enabled', 'active', 'running']
            print(f"[WebSocket] 机器人激活状态: {activate}, 已使能: {is_enabled}")
            return is_enabled
        except Exception as e:
            print(f"[WebSocket] 检查使能状态失败: {e}")
            return False

    def init_follower_mode(self, login_first: bool = True, skip_enable: bool = False) -> bool:
        """
        初始化跟随模式

        执行顺序:
        1. logout/login - 登录
        2. manual_en - 使能机器人 (必须在 start_follower 之前)
        3. start_follower - 开启跟随功能
        4. set_jog_coordinate --tool - 切换到笛卡尔坐标系
        5. follower_cart - 启动笛卡尔跟随模式

        Args:
            login_first: 是否先登录 (默认 True)
            skip_enable: 是否跳过使能步骤 (如果已通过示教器使能，设为 True)

        Returns:
            是否初始化成功
        """
        print("[WebSocket] 开始初始化跟随模式...")

        # 1. 登录
        if login_first:
            self.send_command("logout")
            time.sleep(0.5)

            pwd_md5 = hashlib.md5("000000".encode()).hexdigest()
            success, response = self.send_command(f"login --user=Engineer --pwd={pwd_md5}")
            if not success:
                print("[WebSocket] 登录失败")
                return False
            time.sleep(0.5)

        # 获取当前状态，检查是否有错误
        self.send_get_status()
        time.sleep(0.3)

        if self.last_status:
            try:
                ret_context = self.last_status.get('ret_context', {})
                if isinstance(ret_context, str):
                    ret_context = json.loads(ret_context)
                robot_msg = ret_context.get('robot_msg', {})
                status = robot_msg.get('status', '')
                if status == 'Error':
                    print("[WebSocket] 检测到错误状态，尝试清除...")
                    self.send_command("cl")
                    time.sleep(0.5)
            except:
                pass

        # 2. 使能机器人 (必须在 start_follower 之前!)
        if not skip_enable:
            print("[WebSocket] 使能机器人...")
            success, _ = self.send_command("manual_en")
            if not success:
                print("[WebSocket] manual_en 失败")
                return False
            time.sleep(1.0)

        # 3. 开启跟随功能
        success, _ = self.send_command("start_follower")
        if not success:
            print("[WebSocket] start_follower 失败")
            return False
        time.sleep(1.0)

        # 4. 切换到工具坐标系 (Eye-in-Hand 视觉伺服使用)
        success, _ = self.send_command("set_jog_coordinate --tool")
        if not success:
            print("[WebSocket] set_jog_coordinate --tool 失败")
            return False
        time.sleep(0.5)

        # 5. 启动笛卡尔跟随 (不等待响应，因为这是持续运行的命令)
        print("[WebSocket] 启动笛卡尔跟随模式...")
        self._send_command_no_wait("follower_cart")

        # 等待一下让命令生效
        time.sleep(1.0)

        # 检查状态确认跟随模式已启动
        self.send_get_status()
        if self.last_status:
            try:
                ret_context = self.last_status.get('ret_context', {})
                if isinstance(ret_context, str):
                    ret_context = json.loads(ret_context)
                robot_msg = ret_context.get('robot_msg', {})
                motion = robot_msg.get('motion', '')
                if motion == 'Running':
                    print("[WebSocket] 跟随模式初始化完成!")
                    return True
            except:
                pass

        print("[WebSocket] 跟随模式可能已启动，请检查机器人状态")
        return True

    def _send_command_no_wait(self, command: str) -> bool:
        """
        发送命令但不等待响应 (用于持续运行的命令如 follower_cart)
        注：为了维持 WebSocket 协议同步，实际上仍会尝试接收一次响应并丢弃
        """
        if not self.is_connected or not self.ws:
            return False

        with self._sock_lock:
            try:
                import websocket
                payload = command.encode('utf-8')
                packet = self._pack_header(len(payload)) + payload
                self.ws.send(packet, opcode=websocket.ABNF.OPCODE_BINARY)
                
                # 尝试接收响应以保持同步 (设置极短超时避免阻塞)
                # 对于某些指令，如果已知没有回包可以跳过，但通常都有
                try:
                    self.ws.settimeout(0.1)
                    self.ws.recv()
                    self.ws.settimeout(self.timeout)
                except:
                    pass
                
                # print(f"[WebSocket] 已发送指令 '{command}' (异步模式)")
                return True
            except Exception as e:
                print(f"[WebSocket] 发送指令 '{command}' 失败: {e}")
                self.is_connected = False
                return False

    def stop_follower_mode(self) -> bool:
        """
        停止跟随模式

        Returns:
            是否停止成功
        """
        print("[WebSocket] 停止跟随模式...")
        success, _ = self.send_command("stop_follower")
        return success

    def enable_robot(self, virtual_mode: bool = True) -> bool:
        """
        使能机器人

        Args:
            virtual_mode: True 使用虚拟模式 (manual_en), False 使用真机模式 (en)

        Returns:
            是否使能成功
        """
        cmd = "manual_en" if virtual_mode else "en"
        print(f"[WebSocket] 使能机器人 ({cmd})...")
        success, _ = self.send_command(cmd)
        return success

    def disable_robot(self, virtual_mode: bool = True) -> bool:
        """
        掉使能机器人

        Args:
            virtual_mode: True 使用虚拟模式 (manual_ds), False 使用真机模式 (ds)

        Returns:
            是否成功
        """
        cmd = "manual_ds" if virtual_mode else "ds"
        print(f"[WebSocket] 掉使能机器人 ({cmd})...")
        success, _ = self.send_command(cmd)
        return success

    def set_velocity(self, percent: int = 60, manual_mode: bool = True) -> bool:
        """
        设置速度百分比

        Args:
            percent: 速度百分比 (1-100)
            manual_mode: True 设置手动模式速度, False 设置自动模式速度

        Returns:
            是否设置成功
        """
        if manual_mode:
            cmd = f"set_jog_vel --vel_percent={percent}"
        else:
            cmd = f"set_pgm_vel --vel_percent={percent}"
        print(f"[WebSocket] 设置速度: {percent}%")
        success, _ = self.send_command(cmd)
        return success

    def login(self, user: str = "Engineer", password: str = "000000") -> bool:
        """
        登录机器人控制器

        Args:
            user: 用户名
            password: 密码 (明文，会自动 MD5 加密)

        Returns:
            是否登录成功
        """
        pwd_md5 = hashlib.md5(password.encode()).hexdigest()
        cmd = f"login --user={user} --pwd={pwd_md5}"
        print(f"[WebSocket] 登录: {user}")
        success, _ = self.send_command(cmd)
        return success

    def logout(self) -> bool:
        """登出"""
        success, _ = self.send_command("logout")
        return success

    def send_get_status(self) -> bool:
        """
        发送 get 指令获取机器人状态

        Returns:
            是否成功获取状态
        """
        if not self.is_connected or not self.ws:
            return False

        with self._sock_lock:
            try:
                import websocket

                # 发送 get 指令
                payload = b"get"
                packet = self._pack_header(len(payload)) + payload
                self.ws.send(packet, opcode=websocket.ABNF.OPCODE_BINARY)

                # 接收响应
                resp = self.ws.recv()
                if len(resp) > 40:
                    self._parse_robot_state(resp[40:])
                    return True
                return False

            except Exception as e:
                # print(f"[WebSocket] get 指令失败: {e}")
                self.is_connected = False
                return False

    def _parse_robot_state(self, data: bytes):
        """解析机器人状态数据"""
        try:
            response = json.loads(data.decode('utf-8'))

            # 保存完整状态
            self.last_status = response

            if response.get('ret_code', -1) != 0:
                return

            ret_context = response.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)

            motion_msg = ret_context.get('motion_msg', {})

            with self._position_lock:
                # 提取末端位姿 (欧拉角)
                pe = motion_msg.get('pe', [])
                if pe and len(pe) > 0 and len(pe[0]) >= 6:
                    # 工控机返回的旋转顺序是 [rz, ry, rx]，需要转换为 [rx, ry, rz]
                    pe_raw = pe[0][:6]
                    self._actual_pe = np.array([
                        pe_raw[0], pe_raw[1], pe_raw[2],  # x, y, z
                        pe_raw[5], pe_raw[4], pe_raw[3]   # rx, ry, rz (交换顺序)
                    ], dtype=np.float64)

                # 提取末端位姿 (四元数)
                pq = motion_msg.get('pq', [])
                if pq and len(pq) > 0 and len(pq[0]) >= 7:
                    self._actual_pq = np.array(pq[0][:7], dtype=np.float64)

                # 提取关节角度
                actual_pos = motion_msg.get('actual_pos', [])
                if actual_pos and len(actual_pos) > 0 and len(actual_pos[0]) >= 6:
                    self._actual_joint_pos = np.array(actual_pos[0][:6], dtype=np.float64)

        except Exception as e:
            pass

    def start_polling(self, interval: float = 0.1):
        """
        启动后台轮询线程，定期获取机器人状态

        Args:
            interval: 轮询间隔 (秒)
        """
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, args=(interval,), daemon=True)
        self._poll_thread.start()
        print(f"[WebSocket] 状态轮询已启动，间隔 {interval}s")

    def _poll_loop(self, interval: float):
        """后台轮询循环"""
        while self._running and self.is_connected:
            try:
                self.send_get_status()
            except:
                pass
            time.sleep(interval)

    def get_actual_position_m(self) -> Optional[np.ndarray]:
        """
        获取机器人末端实际位置 (米)

        Returns:
            [x, y, z] (单位: 米) 或 None
        """
        with self._position_lock:
            if self._actual_pe is not None:
                # pe 返回的位置单位是毫米，转换为米
                pos = self._actual_pe[:3].copy() / 1000.0
                return pos
        return None

    def get_actual_pose_m(self) -> Optional[np.ndarray]:
        """
        获取机器人末端实际位姿 (米, 弧度)

        Returns:
            [x, y, z, rx, ry, rz] (单位: 米, 弧度) 或 None
        """
        with self._position_lock:
            if self._actual_pe is not None:
                pose = self._actual_pe.copy()
                # 位置: 毫米 -> 米
                pose[:3] = pose[:3] / 1000.0
                # 角度: 度 -> 弧度
                pose[3:] = np.deg2rad(pose[3:])
                return pose
        return None

    def get_actual_joint_pos(self) -> Optional[np.ndarray]:
        """
        获取机器人关节实际角度

        Returns:
            [j1, j2, j3, j4, j5, j6] (单位: 度) 或 None
        """
        with self._position_lock:
            if self._actual_joint_pos is not None:
                return self._actual_joint_pos.copy()
        return None


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    import math

    # 测试 UDP 客户端
    client = FollowerClient("192.168.0.10", 9998)

    if client.connect():
        print("\n--- 测试发送位姿指令 ---")

        # 发送一个小的增量位移
        # 注意：这里的值是相对位移，单位是米
        success = client.send_pose_euler(
            x=0.01,   # 10mm
            y=0.0,
            z=0.0,
            rx=0.0,
            ry=0.0,
            rz=0.0
        )

        if success:
            print("位姿指令已发送")
            time.sleep(0.5)

            # 检查响应
            resp = client.get_last_response()
            if resp:
                print(f"收到响应: {resp}")

        client.close()
    else:
        print("连接失败，请检查网络和端口配置")
