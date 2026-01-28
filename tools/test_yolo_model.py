# test_yolo_model.py
"""
YOLOv5s 模型权重测试脚本
用于测试 model/best.pt 的检测效果

使用方法:
    pip install ultralytics
    python tools/test_yolo_model.py --source camera_1/data/circle_test.mp4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[错误] 请先安装 ultralytics: pip install ultralytics")
    sys.exit(1)


class YOLOv5Tester:
    def __init__(self, weights_path='model/best.pt', conf_threshold=0.5, iou_threshold=0.45):
        """
        初始化 YOLOv5 测试器
        :param weights_path: 模型权重路径
        :param conf_threshold: 置信度阈值
        :param iou_threshold: NMS IOU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载模型 (使用 ultralytics 本地加载，无需网络)
        print(f"[YOLOv5Tester] 正在加载模型: {weights_path}")
        self.model = YOLO(weights_path)

        # 获取类别名称
        self.class_names = self.model.names
        print(f"[YOLOv5Tester] 模型加载成功")
        print(f"[YOLOv5Tester] 检测类别: {self.class_names}")

    def detect_image(self, image):
        """
        对单张图像进行检测
        :param image: BGR 格式的图像 (numpy array)
        :return: 检测结果
        """
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        return results

    def draw_results(self, image, results):
        """
        在图像上绘制检测结果 (只显示置信度最高的目标，并标注圆心)
        :param image: 原始图像
        :param results: 检测结果
        :return: (绘制后的图像, 圆心坐标或None)
        """
        img_copy = image.copy()
        center = None

        # 获取检测框 (ultralytics 格式)
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # 找到置信度最高的检测框
            confs = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)

            # 获取最佳检测框的坐标
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 获取置信度和类别
            conf = float(confs[best_idx])
            cls = int(boxes.cls[best_idx].cpu().numpy())

            # 计算圆心 (边界框中心)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # 绘制圆心 (红色实心圆)
            cv2.circle(img_copy, center, 8, (0, 0, 255), -1)
            # 绘制圆心十字线
            cv2.line(img_copy, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
            cv2.line(img_copy, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)

            # 绘制标签
            label = f"{self.class_names[cls]}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 显示圆心坐标
            coord_text = f"Center: ({center_x}, {center_y})"
            cv2.putText(img_copy, coord_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return img_copy, center

    def test_on_image(self, image_path):
        """
        测试单张图片
        :param image_path: 图片路径
        """
        print(f"\n[测试图片] {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[错误] 无法读取图片: {image_path}")
            return

        results = self.detect_image(image)
        print(results)

        # 绘制并显示结果
        img_result, center = self.draw_results(image, results)
        if center:
            print(f"[圆心坐标] {center}")
        cv2.imshow('YOLOv5 Detection', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_on_video(self, video_path, save_output=False):
        """
        测试视频文件
        :param video_path: 视频路径
        :param save_output: 是否保存输出视频
        """
        print(f"\n[测试视频] {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[错误] 无法打开视频: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[视频信息] {width}x{height}, {fps:.2f} FPS, {total_frames} 帧")

        # 输出视频写入器
        writer = None
        if save_output:
            output_path = str(Path(video_path).stem) + '_yolo_result.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[输出] 保存到: {output_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            results = self.detect_image(frame)
            img_result, center = self.draw_results(frame, results)

            # 显示帧信息
            info_text = f"Frame: {frame_idx}/{total_frames}"
            cv2.putText(img_result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 显示检测数量
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            det_text = f"Detections: {num_detections}"
            cv2.putText(img_result, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('YOLOv5 Detection', img_result)

            if writer:
                writer.write(img_result)

            # 按 'q' 退出, 按空格暂停
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[完成] 处理了 {frame_idx} 帧")

    def test_on_camera(self, camera_id=0):
        """
        测试实时摄像头
        :param camera_id: 摄像头 ID
        """
        print(f"\n[测试摄像头] ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头: {camera_id}")
            return

        print("[提示] 按 'q' 退出, 按 's' 截图保存")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            results = self.detect_image(frame)
            img_result, center = self.draw_results(frame, results)

            # 显示检测数量
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            info_text = f"Detections: {num_detections}"
            cv2.putText(img_result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('YOLOv5 Detection - Camera', img_result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f'camera_capture_{frame_count}.jpg'
                cv2.imwrite(save_path, img_result)
                print(f"[截图] 已保存: {save_path}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv5 模型测试工具')
    parser.add_argument('--weights', type=str, default='model/best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default='0', help='输入源: 图片路径/视频路径/摄像头ID')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU 阈值')
    parser.add_argument('--save', action='store_true', help='保存输出视频')

    args = parser.parse_args()

    # 初始化测试器
    tester = YOLOv5Tester(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    source = args.source

    # 判断输入类型
    if source.isdigit():
        # 摄像头
        tester.test_on_camera(int(source))
    elif os.path.isfile(source):
        # 文件
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            tester.test_on_image(source)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            tester.test_on_video(source, save_output=args.save)
        else:
            print(f"[错误] 不支持的文件格式: {ext}")
    else:
        print(f"[错误] 无效的输入源: {source}")


if __name__ == '__main__':
    main()
