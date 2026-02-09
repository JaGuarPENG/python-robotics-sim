"""
视觉轨迹提取 (高度修正版)
"""
import cv2
import numpy as np
import os
import sys
import pandas as pd
from ultralytics import YOLO

# 引入项目配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 从 config 同步参数
WEIGHTS = config.YOLO_WEIGHTS_PATH
SOURCE = config.VISION_SOURCE
CALIB_FILE = config.CAMERA_CALIBRATION_FILE
OUTPUT_CSV = 'csv/vision_trajectory.csv'
OFFSET = config.VISION_TO_ROBOT_OFFSET

def main():
    print(f"=== 视觉轨迹提取 (高度修正版 Z=200) ===")
    model = YOLO(WEIGHTS)
    cam_mtx = np.load(CALIB_FILE)['camera_matrix']
    cap = cv2.VideoCapture(SOURCE)
    
    trajectory_points = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, conf=0.6, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy.cpu().numpy()[0]
            cx, cy = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
            
            # 1. 像素转机器人坐标 (m) - 此时 z_h = 0.20
            z_h = OFFSET[2]
            x_robot = (cx - cam_mtx[0,2]) * z_h / cam_mtx[0,0] + OFFSET[0]
            y_robot = (cy - cam_mtx[1,2]) * z_h / cam_mtx[1,1] + OFFSET[1]
            
            # 2. 存储轨迹点 (Z=200, Rx=180)
            trajectory_points.append([
                x_robot * 1000.0, 
                y_robot * 1000.0, 
                z_h * 1000.0, 
                180, 0, 0
            ])
            
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        
        cv2.imshow("Vision Trajectory Extraction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

    if len(trajectory_points) > 0:
        df = pd.DataFrame(trajectory_points, columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
        os.makedirs('csv', exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[成功] 轨迹已保存 (Z = {OFFSET[2]*1000}mm)")

if __name__ == "__main__":
    main()
