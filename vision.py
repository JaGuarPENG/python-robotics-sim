# vision_system.py
import numpy as np
import cv2
import config
from tools.calibrate_camera import MarkerTracker

class VisionSystem:
    def __init__(self, source=config.VISION_SOURCE):
        self.source = source
        self.tracker = MarkerTracker()
        
        # 加载标定参数
        if not self.tracker.load_calibration(config.CAMERA_CALIBRATION_FILE):
            print(f"[警告] 无法加载标定文件 {config.CAMERA_CALIBRATION_FILE}")
            
        self.tracker.marker_size = config.ARUCO_MARKER_SIZE
            
        # 处理输入源
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视觉源: {source}")
        
        # 获取视频元数据
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # 状态记录
        self.current_frame_idx = -1
        self.latest_pos = None      # 缓存上一次有效位置
        self.latest_valid = False
        
        print(f"[VisionSystem] 已加载源: {source}")
        print(f"               时长: {self.duration:.2f}s, FPS: {self.fps:.2f}, 总帧数: {self.total_frames}")

    def get_duration(self):
        return self.duration

    def update_to_time(self, sim_time):
        """
        同步方法：根据仿真时间跳转到对应的视频帧进行识别
        :param sim_time: 当前仿真时间 (秒)
        :return: (position_array, is_valid)
        """
        # 如果是实时相机(FPS=0或非常大)，总是读最新帧
        if self.fps <= 0:
            return self._read_and_detect()

        # 1. 计算目标帧号
        target_frame_idx = int(sim_time * self.fps)
        
        # 2. 边界检查
        if target_frame_idx >= self.total_frames:
            print(f"[VisionSystem] 视频结束 (请求帧 {target_frame_idx} > 总帧 {self.total_frames})")
            return None, False
            
        # 3. 帧同步逻辑
        if target_frame_idx == self.current_frame_idx:
            # 时间还没走到下一帧，返回缓存结果
            return self.latest_pos, self.latest_valid
        
        elif target_frame_idx == self.current_frame_idx + 1:
            # 刚好是下一帧，直接读取 (最快)
            pass 
        else:
            # 跳帧了（比如仿真步长很大），需要 seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        
        # 4. 读取与识别
        ret, frame = self.cap.read()
        self.current_frame_idx = target_frame_idx
        
        if not ret:
            self.latest_valid = False
            return None, False
            
        return self._detect_in_frame(frame)

    def _read_and_detect(self):
        """仅用于实时相机的读取"""
        ret, frame = self.cap.read()
        if not ret: return None, False
        return self._detect_in_frame(frame)

    def _detect_in_frame(self, frame):
        """在单帧图像中进行检测的核心逻辑"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.tracker.detector.detectMarkers(gray)
        
        valid_detection = False
        detected_pos = None
        
        if ids is not None and len(ids) > 0:
            # 默认跟踪第一个识别到的 ID
            idx = 0 
            
            # 获取角点并计算中心
            corners_2d = corners[idx][0]
            center_x = np.mean(corners_2d[:, 0])
            center_y = np.mean(corners_2d[:, 1])
            center_point = np.array([[center_x, center_y]], dtype=np.float32)

            # 坐标转换：像素 -> 世界坐标 (mm)
            if self.tracker.camera_matrix is not None:
                world_points, _ = self.tracker.pixel_to_world(center_point, corners_2d)
                
                if world_points is not None and len(world_points) > 0:
                    # 原始输出 (mm) -> 转换为 (m)
                    wx, wy, wz = world_points[0]
                    x_m = wx / 1000.0
                    y_m = wy / 1000.0
                    z_m = wz / 1000.0
                    
                    # 应用坐标系偏移 (Vision -> Robot Base)
                    x_final = x_m + config.VISION_TO_ROBOT_OFFSET[0]
                    y_final = y_m + config.VISION_TO_ROBOT_OFFSET[1]
                    z_final = z_m + config.VISION_TO_ROBOT_OFFSET[2]
                    
                    detected_pos = np.array([x_final, y_final, z_final])
                    valid_detection = True
        
        # 更新缓存
        if valid_detection:
            self.latest_pos = detected_pos
            self.latest_valid = True
        else:
            # 如果这一帧没识别到，这里有两种策略：
            # 1. 保持上一帧的值 (True) -> 适合短暂遮挡
            # 2. 标记为无效 (False) -> 交给 KF 纯预测
            # 这里我们选择标记为无效，让 KF 去处理丢帧
            self.latest_valid = False
            # self.latest_pos 保持不变或设为 None，取决于你想不想画图时看到残留
            
        return self.latest_pos, self.latest_valid

    def stop(self):
        if self.cap:
            self.cap.release()