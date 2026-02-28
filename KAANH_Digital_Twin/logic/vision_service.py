"""
视觉检测服务 (客户端代理版)
负责：通过网络连接到 vision_server 获取结果
"""

import socket
import json
import numpy as np

class VisionService:
    def __init__(self, config):
        self.config = config
        self.server_host = '127.0.0.1'
        self.server_port = 6000
        
        # 兼容旧接口的属性
        self.last_robot_pos = None

    def capture_and_detect(self):
        """
        向视觉服务端请求最新的圆心机器人坐标
        """
        try:
            # 创建一个短连接请求
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.server_host, self.server_port))
                s.sendall(b"get_pos")
                data = s.recv(1024).decode()
                
                result = json.loads(data)
                if result.get("status") == "ok":
                    pos = result.get("pos")
                    self.last_robot_pos = np.array(pos)
                    return self.last_robot_pos, "识别成功"
                else:
                    return None, result.get("msg", "识别失败")
        except ConnectionRefusedError:
            return None, "无法连接到视觉服务器 (请先运行 tools/vision_server.py)"
        except Exception as e:
            return None, f"通讯错误: {str(e)}"

    def detect_circle_center(self, frame):
        """为了兼容性保留，但客户端模式下不再直接处理图像帧"""
        return None
