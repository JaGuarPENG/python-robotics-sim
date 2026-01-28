# vision_system.py
import numpy as np
import cv2
import config
from tools.calibrate_camera import MarkerTracker


class YOLODetector:
    """YOLOv5 目标检测器"""
    def __init__(self, weights_path, conf_threshold=0.5, iou_threshold=0.45):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("[错误] 请先安装 ultralytics: pip install ultralytics")

        print(f"[YOLODetector] 正在加载模型: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        print(f"[YOLODetector] 模型加载成功, 检测类别: {self.class_names}")

    def detect(self, frame):
        """
        检测图像中的目标，返回置信度最高的目标中心点
        :param frame: BGR 图像
        :return: (center_x, center_y, bbox, confidence) 或 (None, None, None, None)
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # 找到置信度最高的检测框
            confs = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)

            # 获取边界框坐标
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()

            # 计算中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            return center_x, center_y, (x1, y1, x2, y2), float(confs[best_idx])

        return None, None, None, None


class VisionSystem:
    def __init__(self, source=config.VISION_SOURCE, detection_mode=None):
        """
        初始化视觉系统
        :param source: 视频源 (文件路径或相机ID)
        :param detection_mode: 检测模式 ('aruco' 或 'yolo')，默认从 config 读取
        """
        self.source = source
        self.detection_mode = detection_mode or getattr(config, 'DETECTION_MODE', 'aruco')

        # 根据检测模式初始化检测器
        if self.detection_mode == 'yolo':
            self._init_yolo_detector()
        else:
            self._init_aruco_detector()

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
        print(f"               检测模式: {self.detection_mode}")
        print(f"               时长: {self.duration:.2f}s, FPS: {self.fps:.2f}, 总帧数: {self.total_frames}")

    def _init_aruco_detector(self):
        """初始化 ArUco 检测器"""
        self.tracker = MarkerTracker()

        # 加载标定参数
        if not self.tracker.load_calibration(config.CAMERA_CALIBRATION_FILE):
            print(f"[警告] 无法加载标定文件 {config.CAMERA_CALIBRATION_FILE}")

        self.tracker.marker_size = config.ARUCO_MARKER_SIZE
        self.yolo_detector = None
        print(f"[VisionSystem] 使用 ArUco 检测模式")

    def _init_yolo_detector(self):
        """初始化 YOLO 检测器"""
        weights_path = getattr(config, 'YOLO_WEIGHTS_PATH', 'model/best.pt')
        conf_threshold = getattr(config, 'YOLO_CONF_THRESHOLD', 0.5)
        iou_threshold = getattr(config, 'YOLO_IOU_THRESHOLD', 0.45)

        self.yolo_detector = YOLODetector(weights_path, conf_threshold, iou_threshold)
        self.tracker = None

        # YOLO 模式也需要标定参数进行坐标转换
        self._load_calibration_for_yolo()
        print(f"[VisionSystem] 使用 YOLO 检测模式")

    def _load_calibration_for_yolo(self):
        """为 YOLO 模式加载相机标定参数"""
        self.camera_matrix = None
        self.dist_coeffs = None

        try:
            calib_data = np.load(config.CAMERA_CALIBRATION_FILE)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            print(f"[VisionSystem] 已加载标定参数用于 YOLO 坐标转换")
        except Exception as e:
            print(f"[警告] 无法加载标定文件: {e}")
            print(f"[警告] YOLO 模式将使用简化的坐标转换")

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
        if self.detection_mode == 'yolo':
            return self._detect_yolo(frame)
        else:
            return self._detect_aruco(frame)

    def _detect_aruco(self, frame):
        """ArUco 标记检测"""
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

        return self._update_cache(detected_pos, valid_detection)

    def _detect_yolo(self, frame):
        """YOLO 目标检测"""
        center_x, center_y, bbox, conf = self.yolo_detector.detect(frame)

        valid_detection = False
        detected_pos = None

        if center_x is not None:
            # 像素坐标转换为世界坐标
            detected_pos = self._pixel_to_world_yolo(center_x, center_y, bbox)
            if detected_pos is not None:
                valid_detection = True
                # 调试信息：每 50 帧打印一次
                if self.current_frame_idx % 50 == 0:
                    print(f"\r[YOLO] Frame {self.current_frame_idx}: "
                          f"Pixel({center_x:.1f}, {center_y:.1f}) -> "
                          f"World({detected_pos[0]:.3f}, {detected_pos[1]:.3f}, {detected_pos[2]:.3f})m "
                          f"Conf: {conf:.2f}", end="")
        else:
            if self.current_frame_idx % 50 == 0:
                print(f"\r[YOLO] Frame {self.current_frame_idx}: No detection", end="")

        return self._update_cache(detected_pos, valid_detection)

    def _pixel_to_world_yolo(self, center_x, center_y, bbox):
        """
        YOLO 检测结果的像素坐标转世界坐标
        使用简化的平面假设 (假设目标在固定高度平面上)
        """
        if self.camera_matrix is None:
            # 无标定参数时使用简化转换
            # 假设图像中心对应机器人工作空间中心
            img_center_x = 640  # 假设 1280x720 分辨率
            img_center_y = 360

            # 简单的像素到米的转换 (需要根据实际情况调整)
            scale = 0.001  # 1 像素 ≈ 1mm

            x_m = (center_x - img_center_x) * scale
            y_m = (center_y - img_center_y) * scale
            z_m = 0.0
        else:
            # 使用相机标定参数进行转换
            # 假设目标在 Z=0 平面上 (相对于相机)
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            # 假设固定深度 (可以根据 bbox 大小估算)
            # 这里使用配置中的 Z 高度
            z_camera = config.VISION_TO_ROBOT_OFFSET[2] * 1000  # 转为 mm

            # 反投影到 3D
            x_camera = (center_x - cx) * z_camera / fx
            y_camera = (center_y - cy) * z_camera / fy

            # 转换为米
            x_m = x_camera / 1000.0
            y_m = y_camera / 1000.0
            z_m = 0.0

        # 应用坐标系偏移 (Vision -> Robot Base)
        x_final = x_m + config.VISION_TO_ROBOT_OFFSET[0]
        y_final = y_m + config.VISION_TO_ROBOT_OFFSET[1]
        z_final = z_m + config.VISION_TO_ROBOT_OFFSET[2]

        return np.array([x_final, y_final, z_final])

    def _update_cache(self, detected_pos, valid_detection):
        """更新缓存并返回结果"""
        if valid_detection:
            self.latest_pos = detected_pos
            self.latest_valid = True
        else:
            self.latest_valid = False

        return self.latest_pos, self.latest_valid

    def stop(self):
        if self.cap:
            self.cap.release()
