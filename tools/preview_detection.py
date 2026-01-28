"""
YOLO 检测预览工具
同时显示视频画面和 2D 轨迹，用于验证检测和坐标转换是否正确
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DetectionPreview:
    """检测预览器：同时显示视频和轨迹"""

    def __init__(self, video_source=None):
        self.video_source = video_source or config.VISION_SOURCE
        self.cap = None
        self.detector = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # 轨迹存储
        self.pixel_trajectory = []  # 像素坐标轨迹
        self.world_trajectory = []  # 世界坐标轨迹

        # 加载相机标定参数
        self._load_calibration()

    def _load_calibration(self):
        """加载相机标定参数"""
        # 获取项目根目录，确保使用绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        calib_file = os.path.join(project_root, config.CAMERA_CALIBRATION_FILE)
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            print(f"[Preview] 已加载标定文件: {calib_file}")
        else:
            print(f"[Preview] 警告: 标定文件不存在: {calib_file}")
            # 使用默认参数
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros(5)

    def _init_yolo_detector(self):
        """初始化 YOLO 检测器"""
        from ultralytics import YOLO
        model_path = config.YOLO_WEIGHTS_PATH
        print(f"[Preview] 加载 YOLO 模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = config.YOLO_CONF_THRESHOLD
        self.iou_threshold = config.YOLO_IOU_THRESHOLD

    def _detect_yolo(self, frame):
        """YOLO 检测，返回最高置信度的目标"""
        results = self.model(frame, conf=self.conf_threshold,
                            iou=self.iou_threshold, verbose=False)

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)

            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            conf = float(confs[best_idx])

            return center_x, center_y, (x1, y1, x2, y2), conf

        return None, None, None, None

    def _pixel_to_world(self, px, py, z_height=0.1):
        """
        像素坐标转世界坐标 (简化版，假设固定高度平面)
        """
        if self.camera_matrix is None:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # 简化的坐标转换 (假设相机垂直向下看)
        # 实际项目中需要根据相机位姿进行完整转换
        x = (px - cx) * z_height / fx
        y = (py - cy) * z_height / fy
        z = z_height

        return np.array([x, y, z])

    def run(self, max_frames=None, save_output=None):
        """
        运行预览
        :param max_frames: 最大处理帧数 (None 表示处理整个视频)
        :param save_output: 保存输出视频的路径 (None 表示不保存)
        """
        # 初始化
        self._init_yolo_detector()

        # 打开视频
        if isinstance(self.video_source, int):
            self.cap = cv2.VideoCapture(self.video_source)
        else:
            self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            print(f"[Preview] 错误: 无法打开视频源: {self.video_source}")
            return

        # 获取视频信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Preview] 视频信息: {width}x{height}, {fps:.1f}fps, {total_frames} 帧")
        print(f"[Preview] 视频时长: {total_frames/fps:.1f} 秒")

        # 创建显示窗口
        cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2D Trajectory', cv2.WINDOW_NORMAL)

        # 轨迹画布
        trajectory_canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

        # 视频写入器
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_output, fourcc, fps, (width + 500, height))

        frame_idx = 0
        detection_count = 0

        print("\n[Preview] 开始处理... 按 'q' 退出, 按 's' 保存当前轨迹")
        print("-" * 60)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                # YOLO 检测
                cx, cy, bbox, conf = self._detect_yolo(frame)

                # 绘制检测结果
                display_frame = frame.copy()

                if cx is not None:
                    detection_count += 1
                    x1, y1, x2, y2 = bbox

                    # 绘制边界框
                    cv2.rectangle(display_frame, (int(x1), int(y1)),
                                 (int(x2), int(y2)), (0, 255, 0), 2)

                    # 绘制中心点
                    center = (int(cx), int(cy))
                    cv2.circle(display_frame, center, 8, (0, 0, 255), -1)
                    cv2.line(display_frame, (center[0]-15, center[1]),
                            (center[0]+15, center[1]), (0, 0, 255), 2)
                    cv2.line(display_frame, (center[0], center[1]-15),
                            (center[0], center[1]+15), (0, 0, 255), 2)

                    # 显示置信度
                    cv2.putText(display_frame, f"Conf: {conf:.2f}",
                               (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2)

                    # 记录轨迹
                    self.pixel_trajectory.append((cx, cy))

                    # 转换为世界坐标
                    world_pos = self._pixel_to_world(cx, cy)
                    if world_pos is not None:
                        self.world_trajectory.append(world_pos)

                    # 更新轨迹画布
                    self._update_trajectory_canvas(trajectory_canvas)

                # 显示帧信息
                info_text = f"Frame: {frame_idx}/{total_frames} | Detections: {detection_count}"
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 显示像素坐标
                if cx is not None:
                    coord_text = f"Pixel: ({cx:.1f}, {cy:.1f})"
                    cv2.putText(display_frame, coord_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if len(self.world_trajectory) > 0:
                        wp = self.world_trajectory[-1]
                        world_text = f"World: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})m"
                        cv2.putText(display_frame, world_text, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 显示窗口
                cv2.imshow('YOLO Detection', display_frame)
                cv2.imshow('2D Trajectory', trajectory_canvas)

                # 保存合并画面
                if video_writer:
                    # 调整轨迹画布大小以匹配视频高度
                    traj_resized = cv2.resize(trajectory_canvas, (500, height))
                    combined = np.hstack([display_frame, traj_resized])
                    video_writer.write(combined)

                # 进度显示
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100
                    print(f"\r进度: {progress:.1f}% | 帧: {frame_idx} | 检测数: {detection_count}", end="")

                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\n用户退出")
                    break
                elif key == ord('s'):
                    self._save_trajectory()

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n\n用户中断")

        finally:
            self.cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        # 打印统计信息
        print("\n" + "=" * 60)
        print(f"处理完成!")
        print(f"  总帧数: {frame_idx}")
        print(f"  检测成功: {detection_count} ({detection_count/frame_idx*100:.1f}%)")
        print(f"  轨迹点数: {len(self.pixel_trajectory)}")

        if len(self.world_trajectory) > 0:
            traj = np.array(self.world_trajectory)
            print(f"\n世界坐标范围:")
            print(f"  X: [{traj[:,0].min():.3f}, {traj[:,0].max():.3f}] m")
            print(f"  Y: [{traj[:,1].min():.3f}, {traj[:,1].max():.3f}] m")
            print(f"  Z: [{traj[:,2].min():.3f}, {traj[:,2].max():.3f}] m")

        if save_output:
            print(f"\n输出视频已保存: {save_output}")

        return self.pixel_trajectory, self.world_trajectory

    def _update_trajectory_canvas(self, canvas):
        """更新轨迹画布"""
        if len(self.pixel_trajectory) < 2:
            return

        # 清空画布
        canvas[:] = 255

        # 绘制坐标轴
        cv2.line(canvas, (50, 450), (450, 450), (0, 0, 0), 2)  # X轴
        cv2.line(canvas, (50, 450), (50, 50), (0, 0, 0), 2)    # Y轴
        cv2.putText(canvas, "X (pixel)", (400, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(canvas, "Y", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # 计算缩放
        traj = np.array(self.pixel_trajectory)
        x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
        y_min, y_max = traj[:, 1].min(), traj[:, 1].max()

        # 添加边距
        x_range = max(x_max - x_min, 1)
        y_range = max(y_max - y_min, 1)

        # 映射到画布坐标
        def map_to_canvas(px, py):
            cx = int(50 + (px - x_min) / x_range * 380)
            cy = int(450 - (py - y_min) / y_range * 380)
            return (cx, cy)

        # 绘制轨迹
        for i in range(1, len(self.pixel_trajectory)):
            pt1 = map_to_canvas(*self.pixel_trajectory[i-1])
            pt2 = map_to_canvas(*self.pixel_trajectory[i])
            # 颜色渐变：从蓝色到红色
            ratio = i / len(self.pixel_trajectory)
            color = (int(255*(1-ratio)), 0, int(255*ratio))
            cv2.line(canvas, pt1, pt2, color, 2)

        # 绘制起点和终点
        start = map_to_canvas(*self.pixel_trajectory[0])
        end = map_to_canvas(*self.pixel_trajectory[-1])
        cv2.circle(canvas, start, 8, (0, 255, 0), -1)  # 绿色起点
        cv2.circle(canvas, end, 8, (0, 0, 255), -1)    # 红色终点

        # 显示信息
        cv2.putText(canvas, f"Points: {len(self.pixel_trajectory)}",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(canvas, "Green=Start, Red=End",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    def _save_trajectory(self):
        """保存轨迹数据"""
        if len(self.pixel_trajectory) == 0:
            print("\n没有轨迹数据可保存")
            return

        filename = "trajectory_data.npz"
        np.savez(filename,
                 pixel_trajectory=np.array(self.pixel_trajectory),
                 world_trajectory=np.array(self.world_trajectory) if self.world_trajectory else np.array([]))
        print(f"\n轨迹数据已保存: {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO 检测预览工具')
    parser.add_argument('--source', type=str, default=None, help='视频源 (默认使用 config 配置)')
    parser.add_argument('--max-frames', type=int, default=None, help='最大处理帧数')
    parser.add_argument('--save', type=str, default=None, help='保存输出视频路径')
    args = parser.parse_args()

    preview = DetectionPreview(video_source=args.source)
    preview.run(max_frames=args.max_frames, save_output=args.save)


if __name__ == '__main__':
    main()
