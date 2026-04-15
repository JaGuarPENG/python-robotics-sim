# -*- coding: utf-8 -*-
"""
TCP 手动 WebSocket 握手 + ARIS 协议通信脚本 + TCP 转发服务器
适用于 佳安智能 KAANH 工控机
"""

import socket
import struct
import json
import time
import threading


class JointAngleServer:
    """TCP 服务器，将关节角数据发送给连接的客户端"""
    
    def __init__(self, host="0.0.0.0", port=6000):
        self.host = host
        self.port = port
        self.server_sock = None
        self.clients = []  # 连接的客户端列表
        self.running = False
        self.lock = threading.Lock()
        self.current_joints = [0.0] * 6
        
    def start(self):
        """启动 TCP 服务器"""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)
        self.running = True
        
        print(f"[转发服务器] 已启动，监听 {self.host}:{self.port}")
        
        # 启动接受客户端连接的线程
        accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
        accept_thread.start()
        
    def _accept_clients(self):
        """接受客户端连接"""
        while self.running:
            try:
                self.server_sock.settimeout(1.0)
                client_sock, addr = self.server_sock.accept()
                print(f"[转发服务器] 客户端连接: {addr}")
                
                with self.lock:
                    self.clients.append(client_sock)
                    
                # 立即发送当前关节角
                self._send_to_client(client_sock, self.current_joints)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[转发服务器] 接受连接错误: {e}")
                    
    def update_joints(self, joints):
        """更新关节角并发送给所有客户端"""
        self.current_joints = joints
        
        # 发送给所有连接的客户端
        with self.lock:
            disconnected = []
            for client in self.clients:
                if not self._send_to_client(client, joints):
                    disconnected.append(client)
            
            # 移除断开的客户端
            for client in disconnected:
                self.clients.remove(client)
                try:
                    client.close()
                except:
                    pass
                print(f"[转发服务器] 客户端断开，剩余 {len(self.clients)} 个")
    
    def _send_to_client(self, client_sock, joints):
        """发送关节角给指定客户端"""
        try:
            # 格式: j1,j2,j3,j4,j5,j6\n (无中括号，无空格，结尾加\n便于解析)
            data_str = ','.join([f"{j:.4f}" for j in joints]) + '\n'
            data = data_str.encode('utf-8')
            client_sock.send(data)
            return True
        except Exception as e:
            return False
    
    def stop(self):
        """停止服务器"""
        self.running = False
        with self.lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        if self.server_sock:
            self.server_sock.close()
        print("[转发服务器] 已停止")


class ManualWebSocketClient:
    """手动实现 WebSocket 握手的 TCP 客户端"""
    
    def __init__(self, ip="192.168.1.10", port=5999, timeout=5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.connected = False
        
    def connect(self):
        """建立 TCP 连接并完成 WebSocket 握手"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            print(f"[工控机] 已连接到 {self.ip}:{self.port}")
            
            if not self._handshake():
                print("[WebSocket] 握手失败")
                return False
            
            self.connected = True
            print("[WebSocket] 握手成功")
            return True
            
        except Exception as e:
            print(f"[连接错误] {e}")
            return False
    
    def _handshake(self):
        """执行 WebSocket 握手"""
        import base64
        
        key = base64.b64encode(bytes([0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 
                                       0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0])).decode()
        
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {self.ip}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        
        print(f"[握手请求]\n{request}")
        self.sock.send(request.encode())
        
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            response += chunk
        
        response_str = response.decode()
        print(f"[握手响应]\n{response_str}")
        
        if "101 Switching Protocols" in response_str and "Upgrade: websocket" in response_str:
            return True
        return False
    
    def _build_websocket_frame(self, data, opcode=0x02):
        """构建 WebSocket 数据帧"""
        first_byte = 0x80 | opcode
        
        length = len(data)
        if length < 126:
            second_byte = 0x80 | length
            header = struct.pack("!BB", first_byte, second_byte)
        elif length < 65536:
            second_byte = 0x80 | 126
            header = struct.pack("!BBH", first_byte, second_byte, length)
        else:
            second_byte = 0x80 | 127
            header = struct.pack("!BBQ", first_byte, second_byte, length)
        
        mask_key = bytes([0x12, 0x34, 0x56, 0x78])
        
        masked_data = bytearray(data)
        for i in range(len(masked_data)):
            masked_data[i] ^= mask_key[i % 4]
        
        return header + mask_key + bytes(masked_data)
    
    def _parse_websocket_frame(self):
        """解析 WebSocket 数据帧"""
        header = self._recv_exactly(2)
        if not header:
            return None
        
        first_byte, second_byte = struct.unpack("!BB", header)
        opcode = first_byte & 0x0f
        payload_len = second_byte & 0x7f
        
        if payload_len == 126:
            length_bytes = self._recv_exactly(2)
            payload_len = struct.unpack("!H", length_bytes)[0]
        elif payload_len == 127:
            length_bytes = self._recv_exactly(8)
            payload_len = struct.unpack("!Q", length_bytes)[0]
        
        payload = self._recv_exactly(payload_len)
        
        return opcode, payload
    
    def _recv_exactly(self, n):
        """精确接收 n 字节数据"""
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _pack_aris_header(self, msg_len):
        """打包 40 字节 ARIS 协议头"""
        return struct.pack('<IIQqqq', msg_len, 0, 0, 0, 0, 0)
    
    def send_command(self, command):
        """发送 ARIS 命令"""
        if not self.connected:
            return False
        
        try:
            payload = command.encode('utf-8')
            aris_packet = self._pack_aris_header(len(payload)) + payload
            ws_frame = self._build_websocket_frame(aris_packet, opcode=0x02)
            self.sock.send(ws_frame)
            return True
        except Exception as e:
            return False
    
    def recv_response(self, timeout=5.0):
        """接收响应"""
        if not self.connected:
            return None
        
        try:
            self.sock.settimeout(timeout)
            result = self._parse_websocket_frame()
            if not result:
                return None
            
            opcode, payload = result
            
            if len(payload) > 40:
                msg_len = struct.unpack('<I', payload[:4])[0]
                response_body = payload[40:40+msg_len]
                
                try:
                    return json.loads(response_body.decode('utf-8'))
                except:
                    return response_body.decode('utf-8')
            
            return payload.hex()
            
        except socket.timeout:
            return None
        except Exception as e:
            return None
    
    def get_status(self):
        """获取状态"""
        if self.send_command("get"):
            return self.recv_response()
        return None
    
    def close(self):
        """关闭连接"""
        if self.sock:
            close_frame = struct.pack("!BB", 0x88, 0x00)
            try:
                self.sock.send(close_frame)
            except:
                pass
            self.sock.close()
            self.connected = False
            print("[工控机] 连接已关闭")


def extract_joints(response):
    """从响应中提取关节角"""
    try:
        if isinstance(response, dict):
            ret_context = response.get('ret_context', {})
            if isinstance(ret_context, str):
                ret_context = json.loads(ret_context)
            
            motion_msg = ret_context.get('motion_msg', {})
            actual_pos = motion_msg.get('actual_pos', [[]])
            
            if actual_pos and len(actual_pos[0]) >= 6:
                return actual_pos[0][:6]
    except:
        pass
    return None


def main():
    """主函数"""
    # 启动转发服务器（本机 5999 端口）
    server = JointAngleServer(host="0.0.0.0", port=5999)
    server.start()
    
    # 连接工控机（工控机 5999 端口）
    client = ManualWebSocketClient(ip="192.168.1.10", port=5999)
    
    if not client.connect():
        server.stop()
        return
    
    print("\n[开始读取关节角并转发，按 Ctrl+C 停止]")
    print(f"[提示] 其他程序可以连接 localhost:6000 接收关节角数据")
    print(f"{'时间':<12} {'J1':>10} {'J2':>10} {'J3':>10} {'J4':>10} {'J5':>10} {'J6':>10}")
    print("-" * 80)
    
    try:
        while True:
            resp = client.get_status()
            if resp:
                joints = extract_joints(resp)
                if joints:
                    # 转发给所有连接的客户端
                    server.update_joints(joints)
                    
                    # 本地打印
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"{timestamp:<12} {joints[0]:>10.4f} {joints[1]:>10.4f} {joints[2]:>10.4f} "
                          f"{joints[3]:>10.4f} {joints[4]:>10.4f} {joints[5]:>10.4f}")
                else:
                    print("[未能获取关节角数据]")
            time.sleep(0.05)  # 20Hz
            
    except KeyboardInterrupt:
        print("\n[用户停止]")
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
