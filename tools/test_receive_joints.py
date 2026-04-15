# -*- coding: utf-8 -*-
"""
测试接收关节角数据
连接 localhost:5999 接收转发过来的关节角
"""

import socket
import struct
import json
import time


def receive_joints(host="localhost", port=5999):
    """连接并接收关节角数据"""
    
    try:
        # 连接到转发服务器
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print(f"[已连接] {host}:{port}")
        print("[等待接收关节角数据...]\n")
        
        buffer = b""
        while True:
            try:
                # 接收数据到缓冲区
                chunk = sock.recv(1024)
                if not chunk:
                    print("[服务器断开]")
                    return
                
                buffer += chunk
                
                # 按换行符分割处理
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    data_str = line.decode('utf-8')
                    
                    # 解析数据 (格式: j1,j2,j3,j4,j5,j6)
                    joints = [float(x) for x in data_str.split(',')]
                    
                    # 打印关节角
                    print(f"关节角: {data_str}")
                
            except ValueError as e:
                print(f"[解析错误] 收到的数据: {line}, 错误: {e}")
            except Exception as e:
                print(f"[接收错误] {e}")
                break
                
    except ConnectionRefusedError:
        print(f"[连接失败] 无法连接到 {host}:{port}")
        print("[提示] 请先运行 tcp_websocket_client.exe")
    except Exception as e:
        print(f"[错误] {e}")
    finally:
        sock.close()
        print("[连接已关闭]")


if __name__ == "__main__":
    receive_joints()
