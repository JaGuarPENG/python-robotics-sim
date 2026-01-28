# config.py

# --- 真实视觉系统配置 ---
VISION_MODE = True         # 是否使用真实视觉系统 (True: 真实视觉; False: 虚拟轨迹)
VISION_SOURCE = 'camera_1/data/circle_test.mp4'  # 视频文件路径 或 相机ID (如 0)
VISION_UPDATE_INTERVAL = 0.02  # 视觉系统更新目标间隔 (秒)，例如 10ms
# 注意：这只是为了控制处理频率，不代表一定要等待这么久

# --- 检测模式配置 ---
DETECTION_MODE = 'yolo'   # 检测模式: 'aruco' (ArUco标记) 或 'yolo' (YOLOv5目标检测)
YOLO_WEIGHTS_PATH = 'model/best.pt'  # YOLO 模型权重路径
YOLO_CONF_THRESHOLD = 0.7  # YOLO 置信度阈值
YOLO_IOU_THRESHOLD = 0.45  # YOLO NMS IOU 阈值

CAMERA_CALIBRATION_FILE = "camera_1/calib/calibration_data.npz" # 标定文件路径

# Aruco 标记真实尺寸 (mm) - 用于位姿估计
ARUCO_MARKER_SIZE = 95.0  # 标记边长 (毫米)

# 坐标系转换参数 (根据实际情况调整)
# 假设 Vision 输出的是: X右, Y下 (OpenCV)
# 机器人需要的是: X前, Y左 (基座坐标系)
# WORLD_OFFSET 用于将视觉坐标系原点对齐到机器人坐标系原点
# [x_offset, y_offset, z_height]
VISION_TO_ROBOT_OFFSET = [0.3, 0.0, 0.15]

# --- 时间与频率 ---
DT_CTRL = 0.008       # 机器人控制周期 8ms (125Hz)
DT_VISION = 0.02     # 视觉采样周期 20ms (50Hz)

# 仿真时长配置
# 对于 'circle' 和 'linear' 模式：使用 SIM_DURATION
# 对于 'csv' 模式：自动使用 CSV 文件的时长
# 对于真实视觉模式：自动使用视频时长或无限运行（相机）
SIM_DURATION = 10.0   # 生成轨迹模式的仿真时长 (秒)
AUTO_DURATION = True  # 是否自动根据数据源确定时长

# --- 延迟与滞后 ---
SENSE_DELAY = 0.001   # 视觉处理延迟 (秒)
SERVO_ALPHA = 0.4     # 伺服响应滞后系数 (越小滞后越重)
ESTIMATED_SERVO_LAG = 0.02 # 预估的伺服滞后补偿 (秒)

# --- 机器人初始配置 ---
ROBOT_INIT_EE_POS = [0.25, 0.0, 0.2]  # 机器人初始末端位置 [x, y, z] (米)

# --- 轨迹参数 ---
TRAJ_MODE = 'linear'  # 轨迹模式: 'circle', 'linear', 'csv'

# Circle 模式参数
TRAJ_CENTER = [0.4, 0.0, 0.2]
TRAJ_RADIUS = 0.15
TRAJ_OMEGA = 0.8

# Linear 模式参数
TRAJ_INIT_POS = [0.25, 0.0, 0.2]
TRAJ_VELOCITY = 0.01

# CSV 模式参数
TRAJ_CSV_PATH = 'csv/marker_coordinates_kalman.csv'  # CSV 文件路径
TRAJ_Z_HEIGHT = 0.15  # CSV 轨迹的 Z 高度 (米)
TRAJ_WORLD_OFFSET = [0.3, 0.0]  # 世界坐标系偏移 [x, y]

# 跟踪偏置
TRACKING_OFFSET = [0.0, 0.0, 0.1]

# --- 控制器参数 ---
KP = 6.0
KI = 3.0
MAX_STEP = 0.05
INTEGRAL_LIMIT = 0.5

# --- 卡尔曼滤波器参数 ---
KF_PROCESS_NOISE = 0.05    # 过程噪声协方差
KF_MEASUREMENT_NOISE = 0.5 # 测量噪声协方差
KF_VEL_COV = 200.0        # 速度初始协方差
KF_ACC_COV = 200.0        # 加速度初始协方差