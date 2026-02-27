"""
视觉集成测试脚本：用于在不启动示教器的情况下验证圆心识别与坐标转换逻辑
使用方法: python tools/test_vision_integration.py
"""

import cv2
import numpy as np
import os
import sys

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from teach_pendant.logic.vision_service import VisionService

def main():
    print("=== 视觉集成测试启动 ===")
    
    # 1. 初始化视觉服务
    vision = VisionService(config)
    
    # 2. 打开视频源
    source = config.VISION_SOURCE
    print(f"[测试] 正在打开视频源: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源!")
        return

    cv2.namedWindow('Vision Integration Test', cv2.WINDOW_NORMAL)
    print("[提示] 按 'q' 退出，按 '空格' 暂停")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[信息] 视频处理完毕或读取失败")
            break

        frame_count += 1
        
        # 3. 尝试检测圆心 (像素坐标)
        pixel_pos = vision.detect_circle_center(frame)
        
        # 4. 绘图显示
        display_frame = frame.copy()
        
        if pixel_pos:
            cx, cy = int(pixel_pos[0]), int(pixel_pos[1])
            
            # 5. 坐标转换 (机器人空间坐标)
            robot_pos = vision.pixel_to_robot(pixel_pos[0], pixel_pos[1])
            
            # 绘制视觉结果
            cv2.circle(display_frame, (cx, cy), 10, (0, 255, 0), -1) # 绿色圆点
            cv2.line(display_frame, (cx-20, cy), (cx+20, cy), (0, 255, 0), 2)
            cv2.line(display_frame, (cx, cy-20), (cx, cy+20), (0, 255, 0), 2)
            
            # 在画面上标注坐标
            coord_text = f"Pixel: ({cx}, {cy})"
            robot_text = f"Robot: X:{robot_pos[0]:.3f} Y:{robot_pos[1]:.3f} Z:{robot_pos[2]:.3f}"
            
            cv2.putText(display_frame, coord_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, robot_text, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for circle...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 进度显示
        cv2.putText(display_frame, f"Frame: {frame_count}", (20, frame.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Vision Integration Test', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("=== 测试结束 ===")

if __name__ == "__main__":
    main()
