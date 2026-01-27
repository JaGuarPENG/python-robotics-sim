import cv2
import cv2.aruco as aruco
import numpy as np
import os
import argparse
import time
from typing import Callable, Dict, List, Optional, Tuple

class MarkerTracker:
    """
    ArUco标记跟踪器，提供实时坐标输出接口
    """

    def __init__(self):
        self.callbacks: List[Callable[[int, int, float, float], None]] = []
        self.marker_size = 95  # 标记尺寸，单位：mm
        self.camera_matrix = None  # 相机内参矩阵
        self.dist_coeffs = None    # 畸变系数
        self._setup_detector()
        self._setup_marker_world_points()

    def _setup_detector(self):
        """设置ArUco检测器"""
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(ARUCO_DICT, parameters)

    def _setup_marker_world_points(self):
        """设置标记的世界坐标点（假设标记在XY平面上，Z=0）"""
        # ArUco标记的4个角点在世界坐标系中的位置
        # 左上角为原点 (0,0,0)，右上角 (marker_size,0,0)
        # 右下角 (marker_size,marker_size,0)，左下角 (0,marker_size,0)
        self.marker_world_points = np.array([
            [0, 0, 0],                           # 左上角
            [self.marker_size, 0, 0],            # 右上角
            [self.marker_size, self.marker_size, 0], # 右下角
            [0, self.marker_size, 0]             # 左下角
        ], dtype=np.float32)

    def register_callback(self, callback: Callable[[int, int, float, float], None]):
        """
        注册坐标回调函数

        Args:
            callback: 回调函数，参数为 (frame_id, marker_id, center_x, center_y)
        """
        self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[int, int, float, float], None]):
        """取消注册回调函数"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def clear_callbacks(self):
        """清除所有回调函数"""
        self.callbacks.clear()

    def calibrate_camera(self, video_path: str, max_points: int = 50) -> bool:
        """
        使用视频中的ArUco标记进行相机标定

        Args:
            video_path: 视频文件路径

        Returns:
            是否标定成功
        """
        print("开始相机标定...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return False

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"标定视频信息: {width}x{height}, {total_frames}帧")

        # 存储标定数据
        objpoints = []  # 3D世界坐标点
        imgpoints = []  # 2D图像坐标点

        frame_count = 0
        valid_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 检测ArUco标记
                corners, ids, rejected = self.detector.detectMarkers(gray)

                if ids is not None and len(corners) > 0:
                    # 为每个检测到的标记添加标定点
                    for i in range(len(ids)):
                        objpoints.append(self.marker_world_points)
                        imgpoints.append(corners[i][0])
                        valid_frames += 1

                # 显示进度
                if frame_count % 50 == 0:
                    print(f"标定进度: {frame_count}/{total_frames} 帧, 有效检测: {valid_frames}")

        finally:
            cap.release()

        # 执行相机标定
        if valid_frames >= 10:  # 需要至少10个有效检测
            # 如果检测点太多，只使用前max_points个来加快计算速度
            if len(objpoints) > max_points:
                print(f"检测点数量过多 ({len(objpoints)})，使用前 {max_points} 个进行标定...")
                objpoints = objpoints[:max_points]
                imgpoints = imgpoints[:max_points]

            print(f"开始标定计算，使用 {len(objpoints)} 个检测点...")

            try:
                # 确保数据格式正确
                objpoints = np.array(objpoints, dtype=np.float32)
                imgpoints = np.array(imgpoints, dtype=np.float32)

                print(f"[数据验证] 完成")
                print(f"   目标点形状: {objpoints.shape}")
                print(f"   图像点形状: {imgpoints.shape}")
                print(f"   图像尺寸: {gray.shape[::-1]}")

                print("[标定] 正在执行相机标定算法...")
                print("   这可能需要几秒钟时间，请稍候...")

                # 执行标定
                ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )

                print(f"[完成] 标定计算完成，返回值: {ret}")
                print(f"   重投影误差: {ret:.6f}")

                if ret:
                    # 计算重投影误差
                    total_error = 0
                    for i in range(len(objpoints)):
                        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                                        self.camera_matrix, self.dist_coeffs)
                        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        total_error += error

                    mean_error = total_error / len(objpoints)

                    print("[成功] 相机标定成功！")
                    print(f"[质量] 平均重投影误差: {mean_error:.4f} 像素")
                    quality = '优秀' if mean_error < 1.0 else '良好' if mean_error < 2.0 else '一般'
                    print(f"[评估] 标定质量: {quality}")
                    print(f"[内参] 相机矩阵:")
                    print(f"   焦距 (fx, fy): {self.camera_matrix[0,0]:.2f}, {self.camera_matrix[1,1]:.2f}")
                    print(f"   主点 (cx, cy): {self.camera_matrix[0,2]:.2f}, {self.camera_matrix[1,2]:.2f}")
                    print(f"[畸变] 系数: {self.dist_coeffs.ravel()}")

                    # 保存标定结果
                    np.savez('camera_calibration.npz',
                            camera_matrix=self.camera_matrix,
                            dist_coeffs=self.dist_coeffs)
                    print("[保存] 标定结果已保存到 camera_calibration.npz")
                    return True
                else:
                    print("[失败] 标定算法返回失败")
                    print("   可能原因: 数据质量不足或数值计算问题")
                    print("   建议: 检查视频质量，增加标定板运动的多样性")
                    return False

            except Exception as e:
                print(f"[错误] 标定计算过程中出错: {e}")
                print("   这可能是由于数据格式问题或内存不足导致的")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"有效检测帧数太少 ({valid_frames})，需要至少10帧")
            return False

    def load_calibration(self, calibration_file: str = 'calibration_data.npz') -> bool:
        """
        加载相机标定参数

        Args:
            calibration_file: 标定文件路径

        Returns:
            是否加载成功
        """
        try:
            if os.path.exists(calibration_file):
                data = np.load(calibration_file)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                print(f"[成功] 加载相机标定参数从 {calibration_file}")
                print(f"   相机矩阵尺寸: {self.camera_matrix.shape}")
                print(f"   畸变参数数量: {self.dist_coeffs.shape[1]}")
                return True
            else:
                print(f"[错误] 标定文件不存在: {calibration_file}")
                return False
        except Exception as e:
            print(f"[错误] 加载标定参数失败: {e}")
            return False

    def pixel_to_world(self, image_points: np.ndarray, marker_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将图像像素坐标转换为世界坐标

        Args:
            image_points: 图像中的点坐标 (N, 2) - 通常是标记中心点
            marker_corners: 检测到的标记角点 (4, 2)

        Returns:
            世界坐标 (N, 3) 和变换矩阵
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("错误：相机未标定，无法进行坐标转换")
            return image_points.astype(np.float32), None

        try:
            # 计算标记在图像中的像素尺寸
            marker_width_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
            marker_height_pixels = np.linalg.norm(marker_corners[0] - marker_corners[3])

            # 计算像素到毫米的转换比例
            pixel_to_mm_x = self.marker_size / marker_width_pixels
            pixel_to_mm_y = self.marker_size / marker_height_pixels

            # 计算图像中心点（相机光心在图像中的位置）
            cx = self.camera_matrix[0, 2]  # 主点x坐标
            cy = self.camera_matrix[1, 2]  # 主点y坐标

            # 计算图像点相对于图像中心的偏移（像素）
            center_offset_pixels = image_points - np.array([cx, cy])

            # 转换为世界坐标（毫米）
            world_points = np.zeros((len(image_points), 3), dtype=np.float32)
            world_points[:, 0] = center_offset_pixels[:, 0] * pixel_to_mm_x
            world_points[:, 1] = -center_offset_pixels[:, 1] * pixel_to_mm_y  # Y轴翻转
            world_points[:, 2] = 0  # 假设在XY平面上

            # 创建变换矩阵（仅用于接口兼容性）
            transform_matrix = np.eye(4, dtype=np.float32)

            return world_points, transform_matrix

        except Exception as e:
            print(f"坐标转换出错: {e}")
            return image_points.astype(np.float32), None

    def _notify_callbacks(self, frame_id: int, marker_id: int, center_x: float, center_y: float):
        """通知所有回调函数"""
        for callback in self.callbacks:
            try:
                callback(frame_id, marker_id, center_x, center_y)
            except Exception as e:
                print(f"回调函数执行出错: {e}")

    def track_from_video(self, video_path: str, show_video: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从视频文件跟踪标记

        Args:
            video_path: 视频文件路径
            show_video: 是否显示视频窗口

        Returns:
            相机标定参数 (camera_matrix, dist_coeffs)，如果标定失败返回None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None, None

        print(f"开始跟踪视频: {video_path}")
        print(f"注册的回调函数数量: {len(self.callbacks)}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        # 用于相机标定的数据
        obj_points = np.zeros((4, 3), np.float32)
        obj_points[1, 0], obj_points[1, 1] = self.marker_size, 0
        obj_points[2, 0], obj_points[2, 1] = self.marker_size, self.marker_size
        obj_points[3, 0], obj_points[3, 1] = 0, self.marker_size

        objpoints = []
        imgpoints = []
        detection_times = []

        frame_count = 0
        valid_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 检测ArUco标记
                detection_start = time.time()
                corners, ids, rejected = self.detector.detectMarkers(gray)
                detection_time = (time.time() - detection_start) * 1000

                detection_times.append(detection_time)

                if ids is not None and len(corners) > 0:
                    # 绘制检测到的标记
                    aruco.drawDetectedMarkers(frame, corners, ids)

                    # 处理每个检测到的标记
                    for i in range(len(ids)):
                        objpoints.append(obj_points)
                        imgpoints.append(corners[i][0])

                        valid_frames += 1

                        # 计算标记中心位置
                        corners_2d = corners[i][0]
                        center_x = np.mean(corners_2d[:, 0])
                        center_y = np.mean(corners_2d[:, 1])

                        # 如果相机已标定，进行坐标转换
                        if self.camera_matrix is not None:
                            # 将中心点转换为世界坐标
                            center_point = np.array([[center_x, center_y]], dtype=np.float32)
                            world_points, _ = self.pixel_to_world(center_point, corners_2d)

                            if world_points is not None and len(world_points) > 0:
                                world_x, world_y, world_z = world_points[0]
                                # 通过回调函数通知世界坐标
                                self._notify_callbacks(frame_count, int(ids[i][0]), world_x, world_y)
                            else:
                                # 如果转换失败，使用像素坐标
                                self._notify_callbacks(frame_count, int(ids[i][0]), center_x, center_y)
                        else:
                            # 未标定时使用像素坐标
                            self._notify_callbacks(frame_count, int(ids[i][0]), center_x, center_y)


                        # 在帧上绘制标记ID
                        corner = corners[i][0][0].astype(int)
                        cv2.putText(frame, f'ID: {ids[i][0]}', tuple(corner),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 显示进度
                if frame_count % 100 == 0:
                    print(f"处理进度: {frame_count}/{total_frames} 帧, 有效检测: {valid_frames}")

                # 显示视频
                if show_video:
                    cv2.imshow('Marker Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                        break

        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()

        # 统计信息
        if detection_times:
            avg_time = sum(detection_times) / len(detection_times)
            print(f"平均检测时间: {avg_time:.2f} ms")
        print(f"总共处理: {frame_count} 帧, 有效检测: {valid_frames}")

        # 返回相机标定数据（如果需要的话）
        if valid_frames < 10:
            print("警告: 有效检测帧数太少，无法进行准确的相机标定")
            return None, None

        print("跟踪完成")
        return None, None  # 这里可以返回相机标定结果

    def get_marker_coordinates_at_time(self, video_path: str, time_seconds: float, delay_ms: int = 10) -> Optional[Dict[int, Tuple[float, float]]]:
        """
        根据时间获取视频中指定时刻的标记坐标

        Args:
            video_path: 视频文件路径
            time_seconds: 时间（秒）
            delay_ms: 返回结果前的延迟时间（毫秒），默认为10ms

        Returns:
            标记ID到坐标的字典 {marker_id: (x, y)}，如果没有检测到标记返回None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None

        try:
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                print("错误：无法获取视频帧率")
                return None

            # 计算目标帧号
            target_frame = int(time_seconds * fps)

            if target_frame >= total_frames:
                print(f"错误：指定的时间 {time_seconds} 秒超出视频总时长")
                print(f"视频总时长: {total_frames/fps:.2f} 秒")
                return None

            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print(f"错误：无法读取第 {target_frame} 帧")
                return None

            print(f"正在处理时间 {time_seconds:.2f} 秒 (帧号: {target_frame})")

            # 记录检测开始时间
            detection_start = time.time()

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测ArUco标记
            corners, ids, rejected = self.detector.detectMarkers(gray)

            if ids is None or len(corners) == 0:
                print(f"在时间 {time_seconds:.2f} 秒处未检测到任何标记")
                return {}

            # 存储检测结果
            marker_coordinates = {}

            print(f"在时间 {time_seconds:.2f} 秒处检测到 {len(ids)} 个标记:")

            # 处理每个检测到的标记
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                corners_2d = corners[i][0]

                # 计算标记中心位置
                center_x = np.mean(corners_2d[:, 0])
                center_y = np.mean(corners_2d[:, 1])

                # 如果相机已标定，进行坐标转换
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    try:
                        # 将中心点转换为世界坐标
                        center_point = np.array([[center_x, center_y]], dtype=np.float32)
                        world_points, _ = self.pixel_to_world(center_point, corners_2d)

                        if world_points is not None and len(world_points) > 0:
                            world_x, world_y, world_z = world_points[0]
                            marker_coordinates[marker_id] = (world_x, world_y)
                            print(f"  标记ID {marker_id}: 世界坐标 ({world_x:.2f}, {world_y:.2f}) mm")
                        else:
                            marker_coordinates[marker_id] = (center_x, center_y)
                            print(f"  标记ID {marker_id}: 像素坐标 ({center_x:.2f}, {center_y:.2f})")
                    except Exception as e:
                        print(f"  坐标转换失败，使用像素坐标: {e}")
                        marker_coordinates[marker_id] = (center_x, center_y)
                        print(f"  标记ID {marker_id}: 像素坐标 ({center_x:.2f}, {center_y:.2f})")
                else:
                    marker_coordinates[marker_id] = (center_x, center_y)
                    print(f"  标记ID {marker_id}: 像素坐标 ({center_x:.2f}, {center_y:.2f})")

            # 添加延迟返回
            if delay_ms > 0:
                # 计算识别耗时
                detection_end = time.time()
                detection_time = (detection_end - detection_start) * 1000  # 转换为毫秒

                # 计算还需要等待的时间
                remaining_delay = max(0, delay_ms - detection_time)

                if remaining_delay > 0:
                    print(f"识别耗时: {detection_time:.2f} ms, 还需要等待 {remaining_delay:.2f} 毫秒...")
                    time.sleep(remaining_delay / 1000.0)
                else:
                    print(f"识别耗时: {detection_time:.2f} ms, 无需额外等待")
            else:
                # 如果没有设置延迟，也计算并显示识别耗时
                detection_end = time.time()
                detection_time = (detection_end - detection_start) * 1000
                print(f"识别耗时: {detection_time:.2f} ms")  # 转换为秒

            return marker_coordinates

        finally:
            cap.release()

# 示例回调函数
def print_marker_position(frame_id: int, marker_id: int, center_x: float, center_y: float):
    """默认的打印回调函数"""
    print(f"Frame {frame_id} | Marker ID {marker_id} | Position: ({center_x:.2f}, {center_y:.2f})")

def collect_marker_data(frame_id: int, marker_id: int, center_x: float, center_y: float):
    """收集标记数据的回调函数"""
    # 这里可以保存数据到列表、文件或其他存储中
    if not hasattr(collect_marker_data, 'data'):
        collect_marker_data.data = []
    collect_marker_data.data.append((frame_id, marker_id, center_x, center_y))

# 注意：此文件中的命令行参数解析已移除，避免与独立脚本冲突
# 如需命令行使用，请使用 calibrate_only.py 或 receive.py

def query_marker_at_time(video_path: str = 'bdb.mp4', calibration_file: str = 'calibration_data.npz', delay_ms: int = 10):
    """
    根据用户输入的时间查询视频中标记的坐标

    Args:
        video_path: 视频文件路径
        calibration_file: 相机标定文件路径
        delay_ms: 返回结果前的延迟时间（毫秒），默认为10ms
    """
    # 创建标记跟踪器实例
    tracker = MarkerTracker()

    # 尝试加载相机标定参数
    if not tracker.load_calibration(calibration_file):
        print("警告：未找到相机标定文件，将使用像素坐标")
        print("如果需要世界坐标，请先运行相机标定")
        print()

    print("=" * 50)
    print("ArUco标记坐标查询工具")
    print("=" * 50)
    print(f"视频文件: {video_path}")
    print("输入时间（秒）来查询该时刻的标记坐标")
    print("输入 'q' 或 'quit' 退出")
    print("-" * 50)

    while True:
        try:
            # 获取用户输入
            user_input = input("请输入时间（秒）: ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("退出查询工具")
                break

            # 转换输入为浮点数
            time_seconds = float(user_input)

            if time_seconds < 0:
                print("错误：时间不能为负数")
                continue

            # 查询指定时间的标记坐标
            coordinates = tracker.get_marker_coordinates_at_time(video_path, time_seconds, delay_ms)

            if coordinates is None:
                print("查询失败，请检查视频文件和时间参数")
            elif len(coordinates) == 0:
                print(f"在 {time_seconds:.2f} 秒处未检测到任何标记")
            else:
                print(f"\n在 {time_seconds:.2f} 秒处检测到的标记坐标:")
                for marker_id, (x, y) in coordinates.items():
                    coord_type = "世界坐标(mm)" if tracker.camera_matrix is not None else "像素坐标"
                    print(f"  标记ID {marker_id}: ({x:.2f}, {y:.2f}) - {coord_type}")

            print("-" * 50)

        except ValueError:
            print("错误：请输入有效的数字或 'q' 退出")
        except KeyboardInterrupt:
            print("\n用户中断查询")
            break
        except Exception as e:
            print(f"查询过程中出错: {e}")

def calibrate_camera_from_video(video_path, marker_size=95):
	"""
	Calibrate camera using ArUco markers detected in video
	"""

	# Define the 4X4 bit ArUco tag
	ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

	# Create detector parameters
	parameters = aruco.DetectorParameters()
	detector = aruco.ArucoDetector(ARUCO_DICT, parameters)

	# Open video file
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error: Cannot open video file {video_path}")
		return None, None

	print(f"Calibrating camera using video: {video_path}")

	# Get video properties
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	print(f"Video resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

	# Define 3D object points (marker corners in world coordinates)
	obj_points = np.zeros((4, 3), np.float32)
	obj_points[1, 0], obj_points[1, 1] = marker_size, 0
	obj_points[2, 0], obj_points[2, 1] = marker_size, marker_size
	obj_points[3, 0], obj_points[3, 1] = 0, marker_size

	# Lists to store object points and image points
	objpoints = []  # 3D points in real world space
	imgpoints = []  # 2D points in image plane
	detection_times = []  # Store detection times for averaging

	frame_count = 0
	valid_frames = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_count += 1

		# Convert to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect ArUco markers with timing
		detection_start = time.time()
		corners, ids, rejected = detector.detectMarkers(gray)
		detection_time = (time.time() - detection_start) * 1000  # Convert to milliseconds
		detection_times.append(detection_time)
		

		if ids is not None and len(corners) > 0:
			# Draw detected markers
			aruco.drawDetectedMarkers(frame, corners, ids)

			# Add object points and image points for each detected marker
			for i in range(len(ids)):
				objpoints.append(obj_points)
				imgpoints.append(corners[i][0])

				valid_frames += 1

				# Calculate marker center position
				corners_2d = corners[i][0]
				center_x = np.mean(corners_2d[:, 0])
				center_y = np.mean(corners_2d[:, 1])

				# Real-time output marker position
				print(f"Frame {frame_count} | Marker ID {ids[i][0]} | Position: ({center_x:.2f}, {center_y:.2f})")

				# Draw marker ID on frame
				corner = corners[i][0][0].astype(int)
				cv2.putText(frame, f'ID: {ids[i][0]}', tuple(corner),
						   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# Show progress
		if frame_count % 50 == 0:
			print(f"Processed {frame_count}/{total_frames} frames, valid detections: {valid_frames}")

		# Display frame
		cv2.imshow('Calibration - Press ESC to stop', frame)
		if cv2.waitKey(1) & 0xFF == 27:  # ESC key
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        # 运行坐标查询模式
        video_path = sys.argv[2] if len(sys.argv) > 2 else 'bdb.mp4'
        calibration_file = sys.argv[3] if len(sys.argv) > 3 else 'calibration_data.npz'
        delay_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        query_marker_at_time(video_path, calibration_file, delay_ms)
    else:
        # 显示使用说明
        print("使用方法:")
        print("  python calibrate_camera.py --query [视频文件] [标定文件] [延迟毫秒]")
        print("  python calibrate_camera.py --query bdb.mp4 calibration_data.npz 10")
        print()
        print("参数说明:")
        print("  --query: 进入坐标查询模式")
        print("  视频文件: 默认为 bdb.mp4")
        print("  标定文件: 默认为 calibration_data.npz")
        print("  延迟毫秒: 返回结果前的延迟时间，默认为10ms")
        print()
        print("在查询模式中，您可以输入时间（秒）来获取该时刻的标记坐标")

        # 如果没有参数，运行默认的查询模式
        if len(sys.argv) == 1:
            print("\n启动默认查询模式（延迟10ms）...")
            query_marker_at_time()


