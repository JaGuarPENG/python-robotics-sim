"""
视觉服务端 (独立进程运行)
负责：YOLO 检测、坐标转换、TCP 服务响应
"""

import cv2
import numpy as np
import os
import sys
import socket
import json
import threading
import time
from ultralytics import YOLO

# 将项目根目录加入路径以读取 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class VisionServer:
    def __init__(self, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        self.model = None
        self.camera_matrix = None
        
        # 共享状态
        self.last_robot_pos = None
        self.is_running = True
        
        self._load_resources()
        
    def _load_resources(self):
        """加载模型和标定数据"""
        print(f"[VisionServer] 正在加载模型: {config.YOLO_WEIGHTS_PATH}")
        self.model = YOLO(config.YOLO_WEIGHTS_PATH)
        
        print(f"[VisionServer] 正在加载标定: {config.CAMERA_CALIBRATION_FILE}")
        if os.path.exists(config.CAMERA_CALIBRATION_FILE):
            data = np.load(config.CAMERA_CALIBRATION_FILE)
            self.camera_matrix = data['camera_matrix']
        else:
            print("[错误] 找不到标定文件！")

    def pixel_to_robot(self, px, py):
        """像素转机器人坐标逻辑"""
        if self.camera_matrix is None: return None
        
        z_height = config.VISION_TO_ROBOT_OFFSET[2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x_cam = (px - cx) * z_height / fx
        y_cam = (py - cy) * z_height / fy

        return [
            x_cam + config.VISION_TO_ROBOT_OFFSET[0],
            y_cam + config.VISION_TO_ROBOT_OFFSET[1],
            z_height
        ]

    def _tcp_server_loop(self):
        """处理来自示教器的查询请求"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(5)
        print(f"[VisionServer] 坐标查询服务已启动: {self.host}:{self.port}")
        
        while self.is_running:
            try:
                server_sock.settimeout(1.0)
                conn, addr = server_sock.accept()
                data = conn.recv(1024).decode()
                
                if "get_pos" in data:
                    response = {"status": "ok", "pos": self.last_robot_pos}
                    if self.last_robot_pos is None:
                        response = {"status": "error", "msg": "No target detected"}
                    conn.send(json.dumps(response).encode())
                conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[VisionServer] Socket 异常: {e}")

    def run(self):
        """主循环：视频处理"""
        # 启动 TCP 线程
        threading.Thread(target=self._tcp_server_loop, daemon=True).start()
        
        cap = cv2.VideoCapture(config.VISION_SOURCE)
        print(f"[VisionServer] 开始处理视频源: {config.VISION_SOURCE}")
        
        cv2.namedWindow("Vision Server Preview", cv2.WINDOW_NORMAL)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                # 视频循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # YOLO 检测
            results = self.model(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                xyxy = box.xyxy.cpu().numpy()[0]
                cx, cy = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
                
                # 转换并存储最新坐标
                self.last_robot_pos = self.pixel_to_robot(cx, cy)
                
                # 绘图预览
                cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                pos_str = f"X:{self.last_robot_pos[0]:.3f} Y:{self.last_robot_pos[1]:.3f}"
                cv2.putText(frame, pos_str, (int(cx), int(cy)-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.last_robot_pos = None

            cv2.imshow("Vision Server Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    server = VisionServer()
    server.run()
