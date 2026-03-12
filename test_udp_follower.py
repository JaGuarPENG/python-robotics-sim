#!/usr/bin/env python3
"""
UDP 增量跟随模式测试脚本
严格按照标准流程：WebSocket连接 -> 退出/登录 -> 使能 -> 设置速度 -> UDP连接 -> 开启跟随模式 -> UDP发增量
"""

import socket
import struct
import json
import time
import threading
import websocket
import hashlib
import numpy as np
from typing import Optional, Tuple


class UDPFollowerTester:
    """UDP 跟随模式测试器"""
    
    def __init__(self, robot_ip: str = "192.168.0.10"):
        self.robot_ip = robot_ip
        # 1. 端口配置：5888 控制端口, 5999 状态端口
        self.ws_cmd_port = 5888
        self.ws_state_port = 5999
        self.udp_port = 9998  # UDP 端口
        
        # WebSocket 客户端
        self.ws_cmd: Optional[websocket.WebSocket] = None
        self.ws_state: Optional[websocket.WebSocket] = None
        self.ws_connected = False
        
        # UDP 客户端
        self.udp_socket: Optional[socket.socket] = None
        self.udp_connected = False
        
        # 接收线程控制
        self._recv_running = False
        self._recv_thread: Optional[threading.Thread] = None
        
        # 最新接收的工控机状态
        self.latest_feedback = None
        self.feedback_lock = threading.Lock()
        self._new_feedback_event = threading.Event()
        
        # ARIS 头部格式
        self.HEADER_FORMAT = '<IIQqqq'  # msg_len, msg_id, msg_type, reserved1, reserved2, reserved3
        self.HEADER_SIZE = struct.calcsize(self.HEADER_FORMAT)
        
    # ==================== WebSocket 控制部分 ====================
    
    def ws_connect(self) -> bool:
        """1. 连接 WebSocket (端口 5999 和 5888)"""
        try:
            # 连接 5888 控制端口
            self.ws_cmd = websocket.WebSocket()
            self.ws_cmd.settimeout(5.0)
            self.ws_cmd.connect(f"ws://{self.robot_ip}:{self.ws_cmd_port}")
            print(f"[WebSocket] 已连接到控制端口 ws://{self.robot_ip}:{self.ws_cmd_port}")
            
            # 连接 5999 状态端口 (可选，建立连接以符合要求)
            self.ws_state = websocket.WebSocket()
            self.ws_state.settimeout(5.0)
            self.ws_state.connect(f"ws://{self.robot_ip}:{self.ws_state_port}")
            print(f"[WebSocket] 已连接到状态端口 ws://{self.robot_ip}:{self.ws_state_port}")
            
            self.ws_connected = True
            return True
        except Exception as e:
            print(f"[WebSocket] 连接失败: {e}")
            return False
    
    def ws_send_command(self, command: str, need_reply: bool = True) -> Tuple[bool, Optional[dict]]:
        """发送 WebSocket 命令到 5888 控制端口"""
        if not self.ws_connected or not self.ws_cmd:
            return False, None
        
        try:
            # 打包 ARIS 头部
            payload = command.encode('utf-8')
            header = struct.pack(self.HEADER_FORMAT, len(payload), 0, 0, 0, 0, 0)
            packet = header + payload
            
            self.ws_cmd.send(packet, opcode=websocket.ABNF.OPCODE_BINARY)
            
            if not need_reply:
                return True, None
                
            # 接收响应
            resp = self.ws_cmd.recv()
            if len(resp) >= 40:
                data = resp[40:]
                try:
                    response = json.loads(data.decode('utf-8'))
                    return response.get('ret_code', -1) == 0, response
                except json.JSONDecodeError:
                    return False, None
            return False, None
        except Exception as e:
            print(f"[WebSocket] 发送命令 [{command}] 失败: {e}")
            return False, None
    
    def login(self, user: str = "Engineer", password: str = "000000") -> bool:
        """2. 登录（如果有用户登录，先退出）"""
        # 先退出当前用户，防止其他用户占用
        print("[WebSocket] 尝试退出当前已有用户 (logout)...")
        self.ws_send_command("logout", need_reply=True)
        time.sleep(0.5)
        
        # 再进行登录
        pwd_md5 = hashlib.md5(password.encode()).hexdigest()
        success, resp = self.ws_send_command(f"login --user={user} --pwd={pwd_md5}")
        if success:
            print(f"[WebSocket] 登录成功: {user}")
        else:
            print(f"[WebSocket] 登录失败")
        return success
    
    def enable_robot(self) -> bool:
        """3. 使能机器人"""
        print("[WebSocket] 正在使能机器人 (manual_en)...")
        success, resp = self.ws_send_command("manual_en")
        if success:
            print("[WebSocket] 机器人使能成功")
        else:
            print("[WebSocket] 机器人使能失败")
        return success

    def set_speed(self, percent: int = 100) -> bool:
        """4. 设置速度"""
        print(f"[WebSocket] 设置机器运行速度为 {percent}%...")
        success1, _ = self.ws_send_command(f"set_jog_vel --vel_percent={percent}")
        success2, _ = self.ws_send_command(f"set_pgm_vel --vel_percent={percent}")
        if success1 and success2:
            print("[WebSocket] 速度设置成功")
            return True
        print("[WebSocket] 速度设置失败")
        return False
        
    def udp_connect(self) -> bool:
        """5. 连接 UDP"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.settimeout(2.0)
            self.udp_socket.connect((self.robot_ip, self.udp_port))
            self.udp_connected = True
            
            # 启动 UDP 接收线程
            self._recv_running = True
            self._recv_thread = threading.Thread(target=self._udp_recv_loop, daemon=True)
            self._recv_thread.start()
            
            print(f"[UDP] 5. 已连接到 UDP {self.robot_ip}:{self.udp_port}")
            return True
        except Exception as e:
            print(f"[UDP] 5. 连接 UDP 失败: {e}")
            return False

    def start_follower_mode(self) -> bool:
        """6. 发送一系列跟随指令序列"""
        print("\n[初始化] 6. 开始发送一系列跟随指令...")
        
        # 6.1 start_follower
        success, _ = self.ws_send_command("start_follower")
        if not success:
            print("[错误] start_follower 失败")
            return False
        print("[初始化] (1) start_follower 成功")
        time.sleep(1.0)
        
        # 6.2 set_jog_coordinate --tool
        success, _ = self.ws_send_command("set_jog_coordinate --tool")
        if not success:
            print("[错误] set_jog_coordinate 失败")
            return False
        print("[初始化] (2) 坐标系已切换为 tool (set_jog_coordinate --tool) 成功")
        time.sleep(0.5)
        
        # 6.3 follower_cart
        print("[初始化] (3) 启动笛卡尔跟随 follower_cart")
        self.ws_send_command("follower_cart", need_reply=False)
        time.sleep(1.0)
        
        print("[初始化] 跟随指令序列发送完成！进入跟随状态。\n")
        return True
    
    def stop_follower(self) -> bool:
        """停止跟随模式"""
        success, _ = self.ws_send_command("stop_follower")
        if success:
            print("[WebSocket] 跟随模式已停止")
        return success
    
    # ==================== UDP 通信部分 ====================
    
    def _udp_recv_loop(self):
        """UDP 接收循环"""
        while self._recv_running:
            try:
                data, addr = self.udp_socket.recvfrom(4096)
                if len(data) > self.HEADER_SIZE:
                    response_data = data[self.HEADER_SIZE:]
                    try:
                        feedback = json.loads(response_data.decode('utf-8'))
                        with self.feedback_lock:
                            self.latest_feedback = feedback
                        self._new_feedback_event.set()
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except Exception as e:
                if self._recv_running:
                    print(f"[UDP] 接收错误: {e}")
                break
    
    # ==================== 状态获取部分 ====================
    
    def ws_get_state(self) -> Optional[dict]:
        """通过 5999 状态端口获取实时位置"""
        if not self.ws_connected or not self.ws_state:
            return None
        
        try:
            payload = b"get"
            header = struct.pack(self.HEADER_FORMAT, len(payload), 0, 0, 0, 0, 0)
            self.ws_state.send(header + payload, opcode=websocket.ABNF.OPCODE_BINARY)
            
            resp = self.ws_state.recv()
            if len(resp) >= 40:
                data = json.loads(resp[40:].decode('utf-8'))
                if data.get('ret_code', -1) == 0:
                    ret_context = data.get('ret_context', {})
                    if isinstance(ret_context, str):
                        ret_context = json.loads(ret_context)
                    
                    motion_msg = ret_context.get('motion_msg', {})
                    pe = motion_msg.get('pe', [[]])[0]
                    actual_pos = motion_msg.get('actual_pos', [[]])[0]
                    
                    if pe and actual_pos:
                        # 工控机返回的旋转顺序是 [rz, ry, rx]，需要转换为 [rx, ry, rz]
                        # pe[3]=rz, pe[4]=ry, pe[5]=rx
                        tcp_pe_corrected = [
                            pe[0], pe[1], pe[2],  # x, y, z
                            pe[5], pe[4], pe[3]   # rx, ry, rz (注意顺序交换)
                        ]
                        return {
                            'tcp_pe': tcp_pe_corrected,
                            'joints': actual_pos
                        }
            return None
        except Exception as e:
            return None

    def send_pose(self, x: float, y: float, z: float,
                  rx: float = 0, ry: float = 0, rz: float = 0,
                  wait_feedback: bool = False, feedback_timeout: float = 0.05) -> Tuple[bool, Optional[dict]]:
        """
        7. 通过 UDP 发送增量位姿
        返回 (成功标志, 反馈坐标dict) 其中反馈dict含 tcp_pe 和 joints
        """
        if not self.udp_connected or not self.udp_socket:
            print("[UDP] 未连接")
            return False, None

        try:
            msg = {
                "type": "321",
                "pe": [[x, y, z, rx, ry, rz]],
                "pq": []
            }
            payload = json.dumps(msg).encode('utf-8')
            header = struct.pack(self.HEADER_FORMAT, len(payload), 0, 0, 0, 0, 0)

            if wait_feedback:
                self._new_feedback_event.clear()

            self.udp_socket.send(header + payload)

            if wait_feedback:
                got = self._new_feedback_event.wait(timeout=feedback_timeout)
                if got:
                    feedback = self.get_latest_feedback()
                    return True, self.parse_feedback(feedback) if feedback else None
                return True, None

            return True, None
        except Exception as e:
            print(f"[UDP] 发送失败: {e}")
            return False, None
    
    def get_latest_feedback(self) -> Optional[dict]:
        """获取最新反馈"""
        with self.feedback_lock:
            return self.latest_feedback.copy() if self.latest_feedback else None
    
    def parse_feedback(self, feedback: dict) -> Optional[dict]:
        """解析反馈数据，提取实际位姿"""
        try:
            if feedback.get('ret_code', -1) != 0:
                return None
            
            ret_context = feedback.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)
            
            motion_msg = ret_context.get('motion_msg', {})
            
            pe = motion_msg.get('pe', [[]])[0]
            actual_pos = motion_msg.get('actual_pos', [[]])[0]
            
            return {
                'tcp_pe': pe,
                'joints': actual_pos
            }
        except Exception as e:
            return None
    
    # ==================== 测试用例 ====================
    
    def test_1_fixed_position(self, duration: float = 5.0):
        """测试 1：发送固定偏移量"""
        print("=" * 60)
        print("测试 1：发送固定偏移量 (X=50mm)")
        print("=" * 60)
        
        target_x = 0.05
        start_time = time.time()
        count = 0
        
        udp_feedback_count = 0

        while time.time() - start_time < duration:
            ok, fb = self.send_pose(target_x, 0, 0, 0, 0, 0, wait_feedback=True, feedback_timeout=0.05)
            count += 1

            if fb and fb.get('tcp_pe'):
                # UDP 反馈正常
                udp_feedback_count += 1
                source = "UDP"
            else:
                # UDP 无反馈，尝试 WebSocket 获取状态
                fb = self.ws_get_state()
                source = "WS " if fb and fb.get('tcp_pe') else None

            if fb and fb.get('tcp_pe'):
                tcp = fb['tcp_pe']
                joints = fb.get('joints', [])
                j_str = ', '.join(f"{j:.2f}" for j in joints) if joints else 'N/A'
                print(f"[{count:04d}][{source}] 发送增量 X={target_x*1000:.1f}mm | "
                      f"TCP: X={tcp[0]:.3f} Y={tcp[1]:.3f} Z={tcp[2]:.3f} "
                      f"Rx={tcp[3]:.3f} Ry={tcp[4]:.3f} Rz={tcp[5]:.3f} | "
                      f"关节: [{j_str}]")
            else:
                # 打印原始 UDP 反馈供调试
                raw = self.get_latest_feedback()
                if raw:
                    print(f"[{count:04d}] 发送成功，但 parse_feedback 失败，原始数据: {str(raw)[:120]}")
                else:
                    print(f"[{count:04d}] 发送增量 X={target_x*1000:.1f}mm | UDP+WS 均无反馈")

            # 控制发送频率，保持约50Hz (0.02s)
            time.sleep(0.02)

        print(f"测试完成，共发送 {count} 次，其中 UDP 直接反馈 {udp_feedback_count} 次\n")

    def test_coordinate_mapping(self):
        """
        测试坐标映射关系
        依次测试 X+, X-, Y+, Y- 四个方向的移动，确定正确的坐标映射
        """
        print("=" * 60)
        print("坐标映射测试")
        print("将依次测试 X+, X-, Y+, Y- 四个方向的移动")
        print("每个方向移动 50mm，持续 2 秒")
        print("=" * 60)
        
        # 获取初始位置
        fb = self.ws_get_state()
        if not fb or not fb.get('tcp_pe'):
            print("[错误] 无法获取当前位置")
            return False
        
        origin = fb['tcp_pe']
        print(f"\n📍 初始位置: X={origin[0]:.2f}, Y={origin[1]:.2f}, Z={origin[2]:.2f}")
        input("\n按 Enter 开始测试 X+ 方向 (期望: X 增加 50mm)...")
        
        # 定义测试用例: (描述, 期望移动的基座方向, send_pose 参数 x, y)
        test_cases = [
            ("X+ 方向", "X 增加", 0.050, 0),      # 工具X=50mm, 工具Y=0
            ("X- 方向", "X 减少", -0.050, 0),     # 工具X=-50mm, 工具Y=0
            ("Y+ 方向", "Y 增加", 0, 0.050),      # 工具X=0, 工具Y=50mm
            ("Y- 方向", "Y 减少", 0, -0.050),     # 工具X=0, 工具Y=-50mm
        ]
        
        results = []
        
        for desc, expected, send_x, send_y in test_cases:
            print(f"\n{'='*60}")
            print(f"测试: {desc} ({expected})")
            print(f"发送: send_pose({send_x:.3f}, {send_y:.3f}, 0, ...)")
            print(f"{'='*60}")
            
            # 获取测试前位置
            fb = self.ws_get_state()
            if not fb or not fb.get('tcp_pe'):
                print("[错误] 无法获取位置")
                continue
            before = fb['tcp_pe']
            
            # 发送增量 2 秒
            start_time = time.time()
            count = 0
            while time.time() - start_time < 2.0:
                self.send_pose(send_x, send_y, 0, 0, 0, 0, wait_feedback=False)
                count += 1
                time.sleep(0.02)
            
            # 获取测试后位置
            fb = self.ws_get_state()
            if not fb or not fb.get('tcp_pe'):
                print("[错误] 无法获取位置")
                continue
            after = fb['tcp_pe']
            
            delta_x = after[0] - before[0]
            delta_y = after[1] - before[1]
            
            print(f"移动前: X={before[0]:.2f}, Y={before[1]:.2f}")
            print(f"移动后: X={after[0]:.2f}, Y={after[1]:.2f}")
            print(f"变化量: dX={delta_x:+.2f}, dY={delta_y:+.2f}")
            
            results.append({
                'desc': desc,
                'expected': expected,
                'send_x': send_x,
                'send_y': send_y,
                'delta_x': delta_x,
                'delta_y': delta_y
            })
            
            if desc != "Y- 方向":
                input("\n按 Enter 继续下一个测试...")
        
        # 分析结果
        print(f"\n{'='*60}")
        print("测试结果分析")
        print(f"{'='*60}")
        
        for r in results:
            print(f"\n{r['desc']}: send_pose({r['send_x']:.3f}, {r['send_y']:.3f}, ...)")
            print(f"  实际变化: dX={r['delta_x']:+.2f}, dY={r['delta_y']:+.2f}")
            
            # 判断哪个基座坐标变化最大
            if abs(r['delta_x']) > abs(r['delta_y']):
                main_axis = "X"
                main_delta = r['delta_x']
            else:
                main_axis = "Y"
                main_delta = r['delta_y']
            
            direction = "增加" if main_delta > 0 else "减少"
            print(f"  主要影响: {main_axis}轴 {direction} ({main_delta:+.2f}mm)")
        
        # 推断映射关系
        print(f"\n{'='*60}")
        print("坐标映射推断")
        print(f"{'='*60}")
        
        # 根据结果推断映射关系
        x_plus = results[0]  # send_x=0.05, send_y=0
        y_plus = results[2]  # send_x=0, send_y=0.05
        
        # 判断 send_pose 的 x 参数控制哪个基座轴
        if abs(x_plus['delta_x']) > abs(x_plus['delta_y']):
            x_mapping = "send_pose(x,...) 控制基座 X"
        else:
            x_mapping = "send_pose(x,...) 控制基座 Y"
        
        if abs(y_plus['delta_x']) > abs(y_plus['delta_y']):
            y_mapping = "send_pose(...,y,...) 控制基座 X"
        else:
            y_mapping = "send_pose(...,y,...) 控制基座 Y"
        
        print(f"  {x_mapping}")
        print(f"  {y_mapping}")
        
        return results

    def test_single_point_move(self, move_duration: float = 5.0):
        """
        测试 2：单点遥操作移动
        先显示当前坐标，然后让用户输入目标绝对坐标，通过UDP增量模式移动过去
        
        Args:
            move_duration: 移动过程持续时间(秒)
        """
        print("=" * 60)
        print("测试 2：单点遥操作移动")
        print("=" * 60)
        
        # 1. 获取并显示当前位置
        fb = self.ws_get_state()
        if not fb or not fb.get('tcp_pe'):
            print("[错误] 无法获取当前位置")
            return False
        
        current_pos = fb['tcp_pe']
        print(f"\n📍 当前坐标 (XYZRxRyRz):")
        print(f"   X={current_pos[0]:.3f} mm")
        print(f"   Y={current_pos[1]:.3f} mm")
        print(f"   Z={current_pos[2]:.3f} mm")
        print(f"   Rx={current_pos[3]:.3f}°")
        print(f"   Ry={current_pos[4]:.3f}°")
        print(f"   Rz={current_pos[5]:.3f}°")
        print("-" * 60)
        
        # 2. 让用户输入目标坐标
        print("\n请输入目标坐标 (绝对位置，单位：mm/°):")
        try:
            x_input = input(f"  X [{current_pos[0]:.1f}]: ").strip()
            y_input = input(f"  Y [{current_pos[1]:.1f}]: ").strip()
            z_input = input(f"  Z [{current_pos[2]:.1f}]: ").strip()
            rx_input = input(f"  Rx [{current_pos[3]:.1f}]: ").strip()
            ry_input = input(f"  Ry [{current_pos[4]:.1f}]: ").strip()
            rz_input = input(f"  Rz [{current_pos[5]:.1f}]: ").strip()
            
            # 解析输入，如果为空则使用当前值
            target_x = float(x_input) if x_input else current_pos[0]
            target_y = float(y_input) if y_input else current_pos[1]
            target_z = float(z_input) if z_input else current_pos[2]
            target_rx = float(rx_input) if rx_input else current_pos[3]
            target_ry = float(ry_input) if ry_input else current_pos[4]
            target_rz = float(rz_input) if rz_input else current_pos[5]
            
        except ValueError as e:
            print(f"[错误] 输入格式不正确: {e}")
            return False
        
        # 3. 计算偏移量
        offset_x = target_x - current_pos[0]
        offset_y = target_y - current_pos[1]
        offset_z = target_z - current_pos[2]
        offset_rx = target_rx - current_pos[3]
        offset_ry = target_ry - current_pos[4]
        offset_rz = target_rz - current_pos[5]
        
        print(f"\n📋 移动信息:")
        print(f"   当前: X={current_pos[0]:.2f}, Y={current_pos[1]:.2f}, Z={current_pos[2]:.2f}")
        print(f"   目标: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")
        print(f"   偏移: X={offset_x:+.2f}, Y={offset_y:+.2f}, Z={offset_z:+.2f}")
        
        # 4. 确认移动
        confirm = input("\n确认移动? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消移动")
            return False
        
        # 5. 转换到UDP增量坐标系 (m, rad)
        # 根据坐标映射测试结果:
        # - send_pose(x,...) 直接控制基座 X (1:1)
        # - send_pose(...,y,...) 控制基座 Y 但需要取反
        send_x = offset_x / 1000.0          # X 直接对应
        send_y = -offset_y / 1000.0         # Y 需要取反
        send_z = -offset_z / 1000.0         # Z 取反
        
        # 旋转映射: dry(index 3)=ry, drx(index 4)=rx, drz(index 5)=rz
        dry_rad = np.deg2rad(offset_ry)
        drx_rad = np.deg2rad(offset_rx)
        drz_rad = np.deg2rad(offset_rz)
        
        print(f"\n[UDP增量] 发送参数:")
        print(f"          x={send_x:.4f}m (基座X, 直接)")
        print(f"          y={send_y:.4f}m (基座Y, 取反)")
        print(f"          z={send_z:.4f}m (基座Z, 取反)")
        print(f"          旋转: dry={np.rad2deg(dry_rad):.2f}°, drx={np.rad2deg(drx_rad):.2f}°, drz={np.rad2deg(drz_rad):.2f}°")
        print("-" * 60)
        
        # 6. 发送增量指令
        print(f"\n{'帧':<6} | {'发送_x':>10} {'发送_y':>10} {'发送_z':>10} | "
              f"{'机器人X':>10} {'机器人Y':>10} {'机器人Z':>10} | {'误差(mm)':>10}")
        print("-" * 80)
        
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < move_duration:
            t_loop_start = time.perf_counter()
            
            # 直接发送基座坐标偏移
            self.send_pose(send_x, send_y, send_z, dry_rad, drx_rad, drz_rad, wait_feedback=False)
            count += 1
            
            # 获取当前位置用于显示
            if count % 10 == 0:  # 每10帧显示一次
                fb = self.ws_get_state()
                if fb and fb.get('tcp_pe'):
                    tcp = fb['tcp_pe']
                    # 计算当前误差
                    error = np.linalg.norm([
                        tcp[0] - target_x,
                        tcp[1] - target_y, 
                        tcp[2] - target_z
                    ])
                    print(f"{count:<6} | {send_x:>10.4f} {send_y:>10.4f} {send_z:>10.4f} | "
                          f"{tcp[0]:>10.2f} {tcp[1]:>10.2f} {tcp[2]:>10.2f} | {error:>10.2f}")
            
            # 控制发送频率 50Hz
            time.sleep(max(0, 0.02 - (time.perf_counter() - t_loop_start)))
        
        # 7. 最终状态
        fb = self.ws_get_state()
        if fb and fb.get('tcp_pe'):
            tcp = fb['tcp_pe']
            final_error = np.linalg.norm([
                tcp[0] - target_x,
                tcp[1] - target_y,
                tcp[2] - target_z
            ])
            print("-" * 80)
            print(f"\n✅ 移动完成!")
            print(f"   最终位置: X={tcp[0]:.2f}, Y={tcp[1]:.2f}, Z={tcp[2]:.2f}")
            print(f"   目标位置: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")
            print(f"   位置误差: {final_error:.2f}mm")
            print(f"   共发送 {count} 次指令")
        
        return True
    
    def close(self):
        """关闭连接"""
        print("\n[清理] 正在关闭连接...")
        if self.ws_connected:
            try:
                self.stop_follower()
            except:
                pass
        
        self._recv_running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=1.0)
        
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except:
                pass
            print("[UDP] 已关闭")
        
        if self.ws_cmd:
            try:
                self.ws_cmd.close()
            except:
                pass
        if self.ws_state:
            try:
                self.ws_state.close()
            except:
                pass
        print("[WebSocket] 已关闭")


def main():
    """主执行流"""
    import sys
    
    # 填入你机器人的实际IP
    tester = UDPFollowerTester(robot_ip="192.168.0.10") 
    
    try:
        # 1. 建立 WebSocket 连接 (5999 & 5888)
        if not tester.ws_connect():
            return
            
        # 2. 退出已登录用户并重新登录
        if not tester.login():
            return
        time.sleep(0.5)
        
        # 3. 使能机器人
        if not tester.enable_robot():
            return
        time.sleep(0.5)
        
        # 4. 设置速度
        tester.set_speed(100)
        time.sleep(0.5)

        # 5. 连接 UDP (9998)
        if not tester.udp_connect():
            return
        time.sleep(0.5)
        
        # 6. 发送跟随前置指令序列 (start_follower -> set_jog_coordinate -> follower_cart)
        if not tester.start_follower_mode():
            return
        
        print("\n" + "=" * 60)
        print("请选择测试模式:")
        print("1. 固定偏移量测试 (X=50mm)")
        print("2. 单点遥操作移动 (输入目标坐标)")
        print("3. 坐标映射测试 (自动测试X/Y方向)")
        print("=" * 60)
        
        choice = input("请输入选项 (1, 2 或 3): ").strip()
        
        if choice == "1":
            print("\n" + "=" * 60)
            input("准备开始固定偏移量测试，请确认机器人周围无障碍物，按 Enter 继续...")
            print("=" * 60)
            
            # 测试 1: 通过 UDP 发送增量
            tester.test_1_fixed_position(duration=3.0)
            time.sleep(1.0)
            
        elif choice == "2":
            print("\n" + "=" * 60)
            print("单点遥操作移动 - 先显示当前坐标，再输入目标坐标")
            print("=" * 60)
            
            # 直接调用测试函数，它会显示当前坐标并让用户输入目标
            tester.test_single_point_move(move_duration=5.0)
            
        elif choice == "3":
            print("\n" + "=" * 60)
            print("坐标映射测试 - 自动测试各方向")
            print("=" * 60)
            
            # 测试坐标映射
            tester.test_coordinate_mapping()
        else:
            print("无效选项")
            return
        
        print("\n所有测试完成！")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试异常: {e}")
    finally:
        tester.close()


if __name__ == "__main__":
    main()
