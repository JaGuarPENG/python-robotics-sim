# -*- coding: utf-8 -*-
"""
TCP 直连工控机 - 原始指令测试脚本
协议: WebSocket + ARIS 头 (40字节)
端口: 5999 (控制)
"""

import socket
import struct
import json
import hashlib
import time


class TcpRobotClient:
    """TCP 直连机器人控制器"""
    
    def __init__(self, ip="192.168.1.10", port=5999, timeout=5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None
        
    def connect(self):
        """建立 TCP 连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            print(f"[连接成功] {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[连接失败] {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()
            print("[连接已关闭]")
    
    def _pack_header(self, msg_len):
        """打包 40 字节 ARIS 协议头 (小端序)"""
        # <IIQqqq: unsigned int, unsigned int, unsigned long long, signed long long x3
        return struct.pack('<IIQqqq', msg_len, 0, 0, 0, 0, 0)
    
    def _send_recv(self, command):
        """发送命令并接收响应"""
        if not self.sock:
            print("[错误] 未连接")
            return None
        
        try:
            # 打包发送
            payload = command.encode('utf-8')
            header = self._pack_header(len(payload))
            packet = header + payload
            
            # 打印发送的原始数据
            print(f"\n[发送原始数据] {packet.hex()}")
            print(f"[发送原始文本] {packet}")
            
            self.sock.send(packet)
            
            # 接收响应 (先收40字节头)
            resp_header = self.sock.recv(40)
            print(f"[接收响应头] {resp_header.hex()} (长度: {len(resp_header)})")
            
            if len(resp_header) < 40:
                print("[错误] 响应头不完整")
                return None
            
            # 解析响应体长度
            msg_len = struct.unpack('<I', resp_header[:4])[0]
            print(f"[响应体长度] {msg_len}")
            
            # 接收响应体
            resp_body = b""
            while len(resp_body) < msg_len:
                chunk = self.sock.recv(msg_len - len(resp_body))
                if not chunk:
                    break
                resp_body += chunk
            
            print(f"[接收响应体] {resp_body}")
            
            # 解析 JSON
            return json.loads(resp_body.decode('utf-8'))
            
        except Exception as e:
            print(f"[通信错误] {e}")
            return None
    
    def login(self, user="Manufacturer", password="2045"):
        """登录"""
        pwd_md5 = hashlib.md5(password.encode()).hexdigest()
        cmd = f"login --user={user} --pwd={pwd_md5}"
        print(f"\n[指令] {cmd}")
        
        resp = self._send_recv(cmd)
        if resp:
            print(f"[登录响应] {json.dumps(resp, ensure_ascii=False, indent=2)}")
            if resp.get('ret_code') == 0:
                print("[登录成功]")
                return True
        print(f"[登录失败]")
        return False
    
    def enable(self):
        """使能机器人"""
        cmd = "manual_en"
        print(f"\n[指令] {cmd}")
        
        resp = self._send_recv(cmd)
        if resp:
            print(f"[使能响应] {json.dumps(resp, ensure_ascii=False, indent=2)}")
            if resp.get('ret_code') == 0:
                print("[使能成功]")
                return True
        print(f"[使能失败]")
        return False
    
    def get_status(self):
        """获取状态 (关节角)"""
        cmd = "get"
        resp = self._send_recv(cmd)
        
        if not resp or resp.get('ret_code') != 0:
            print(f"[查询失败] {resp}")
            return None
        
        # 解析关节角
        try:
            ret_context = resp.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)
            
            motion_msg = ret_context.get('motion_msg', {})
            actual_pos = motion_msg.get('actual_pos', [[]])
            
            if actual_pos and len(actual_pos[0]) >= 6:
                joints = actual_pos[0][:6]
                return joints
        except Exception as e:
            print(f"[解析错误] {e}")
        
        return None
    
    def print_joints(self, joints, raw_response=None):
        """打印关节角"""
        if joints:
            print(f"[关节角] J1={joints[0]:.4f}, J2={joints[1]:.4f}, J3={joints[2]:.4f}, "
                  f"J4={joints[3]:.4f}, J5={joints[4]:.4f}, J6={joints[5]:.4f}")
        if raw_response:
            print(f"[完整响应] {json.dumps(raw_response, ensure_ascii=False, indent=2)}")


def main():
    """主流程"""
    client = TcpRobotClient(ip="192.168.1.10", port=5999)
    
    # 1. 连接
    if not client.connect():
        return
    
    try:
        # 2. 登录
        if not client.login("Manufacturer", "2045"):
            return
        time.sleep(0.5)
        
        # 3. 使能
        if not client.enable():
            return
        time.sleep(0.5)
        
        # 4. 循环查询关节角
        print("\n[开始查询关节角，按 Ctrl+C 停止]\n")
        try:
            while True:
                resp = client._send_recv("get")
                if resp:
                    # 解析关节角
                    joints = None
                    try:
                        ret_context = resp.get('ret_context', {})
                        if isinstance(ret_context, str):
                            ret_context = json.loads(ret_context)
                        motion_msg = ret_context.get('motion_msg', {})
                        actual_pos = motion_msg.get('actual_pos', [[]])
                        if actual_pos and len(actual_pos[0]) >= 6:
                            joints = actual_pos[0][:6]
                    except:
                        pass
                    client.print_joints(joints, resp)
                time.sleep(0.1)  # 10Hz
        except KeyboardInterrupt:
            print("\n[用户停止]")
    
    finally:
        client.close()


if __name__ == "__main__":
    main()
