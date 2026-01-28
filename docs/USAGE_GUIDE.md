# 视觉伺服仿真系统使用指南

## 目录
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [检测模式](#检测模式)
- [工具使用](#工具使用)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 运行仿真
```bash
python main.py
```

### 2. 基本流程
1. 系统读取 `config.py` 配置
2. 根据 `VISION_MODE` 选择视觉源（视频/虚拟轨迹）
3. 根据 `DETECTION_MODE` 选择检测方式（ArUco/YOLO）
4. 运行视觉伺服仿真
5. 回放仿真结果

---

## 配置说明

配置文件：`config.py`

### 视觉系统配置

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `VISION_MODE` | 视觉模式开关 | `True`: 使用真实视频/相机<br>`False`: 使用虚拟轨迹 |
| `VISION_SOURCE` | 视频源 | `'camera_1/data/circle_test.mp4'` 或 `0`（相机ID） |
| `VISION_UPDATE_INTERVAL` | 视觉更新间隔 | `0.02`（秒） |

### 检测模式配置

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `DETECTION_MODE` | 检测模式 | `'aruco'` 或 `'yolo'` |
| `YOLO_WEIGHTS_PATH` | YOLO 模型路径 | `'model/best.pt'` |
| `YOLO_CONF_THRESHOLD` | YOLO 置信度阈值 | `0.7` |
| `YOLO_IOU_THRESHOLD` | YOLO NMS IOU 阈值 | `0.45` |
| `CAMERA_CALIBRATION_FILE` | 相机标定文件 | `'camera_1/calib/calibration_data.npz'` |

### 仿真时长配置

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `SIM_DURATION` | 虚拟轨迹模式时长 | `10.0`（秒） |
| `AUTO_DURATION` | 自动确定时长 | `True` |

**注意**：当 `VISION_MODE=True` 时，仿真时长自动匹配视频时长。

---

## 检测模式

### ArUco 标记检测
适用于使用 ArUco 标记作为跟踪目标的场景。

```python
# config.py
DETECTION_MODE = 'aruco'
ARUCO_MARKER_SIZE = 95.0  # 标记边长 (mm)
```

### YOLO 目标检测
适用于使用自定义目标（如小球）的场景。

```python
# config.py
DETECTION_MODE = 'yolo'
YOLO_WEIGHTS_PATH = 'model/best.pt'
YOLO_CONF_THRESHOLD = 0.7
```

---

## 工具使用

### 1. YOLO 检测预览工具

**用途**：在运行仿真前，验证 YOLO 检测效果和轨迹转换是否正确。

**文件**：`tools/preview_detection.py`

**使用方法**：
```bash
# 使用默认配置
python tools/preview_detection.py

# 指定视频源
python tools/preview_detection.py --source camera_1/data/circle_test.mp4

# 限制处理帧数（快速测试）
python tools/preview_detection.py --max-frames 500

# 保存输出视频
python tools/preview_detection.py --save output/preview.mp4
```

**功能**：
- 显示两个窗口：YOLO 检测画面 + 2D 轨迹图
- 实时显示检测框、中心点、置信度
- 实时绘制像素坐标轨迹

**快捷键**：
- `q` - 退出
- `s` - 保存轨迹数据到 `trajectory_data.npz`

---

### 2. YOLO 模型测试工具

**用途**：单独测试 YOLO 模型的检测效果。

**文件**：`tools/test_yolo_model.py`

**使用方法**：
```bash
# 测试视频
python tools/test_yolo_model.py --source camera_1/data/circle_test.mp4

# 测试图片
python tools/test_yolo_model.py --source image.jpg

# 测试摄像头
python tools/test_yolo_model.py --source 0

# 保存检测结果视频
python tools/test_yolo_model.py --source video.mp4 --save

# 指定模型和置信度
python tools/test_yolo_model.py --weights model/best.pt --conf 0.5
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 输入源 | `0`（摄像头） |
| `--weights` | 模型权重路径 | `model/best.pt` |
| `--conf` | 置信度阈值 | `0.5` |
| `--iou` | NMS IOU 阈值 | `0.45` |
| `--save` | 保存输出视频 | 否 |

---

### 3. 仿真回放

仿真结束后会自动进入回放模式。

**回放模式**：
- **轻量级模式**（默认）：只显示轨迹点，速度快
- **完整模式**：渲染 UR5 机器人模型，速度慢

**在代码中切换**：
```python
from tools.video_saver import replay

# 轻量级回放（默认）
replay(results, realtime=True, speed_factor=1.0)

# 完整机器人模型回放
replay(results, realtime=True, speed_factor=1.0, lightweight=False)

# 2倍速回放
replay(results, realtime=True, speed_factor=2.0)
```

---

## 常见问题

### Q1: 仿真时长只有 10 秒，不是视频的完整时长？
**原因**：`VISION_MODE = False`，使用的是虚拟轨迹模式。

**解决**：
```python
# config.py
VISION_MODE = True  # 改为 True
```

### Q2: 回放动画非常卡顿？
**原因**：使用了完整机器人模型渲染模式。

**解决**：使用轻量级回放模式（默认已启用）。

### Q3: 标定文件找不到？
**原因**：路径配置错误或文件不存在。

**解决**：检查 `config.py` 中的 `CAMERA_CALIBRATION_FILE` 路径是否正确。

### Q4: YOLO 模型加载失败？
**原因**：模型文件不存在或路径错误。

**解决**：
1. 确认 `model/best.pt` 文件存在
2. 检查 `YOLO_WEIGHTS_PATH` 配置

### Q5: 回放结束后程序卡住？
**原因**：`plt.show()` 等待用户关闭窗口。

**解决**：关闭 matplotlib 窗口（点击 X 按钮）即可继续。

---

## 典型工作流程

```
1. 准备 YOLO 模型
   └── 将训练好的 best.pt 放入 model/ 目录

2. 配置 config.py
   ├── VISION_MODE = True
   ├── VISION_SOURCE = 'your_video.mp4'
   ├── DETECTION_MODE = 'yolo'
   └── CAMERA_CALIBRATION_FILE = 'path/to/calibration.npz'

3. 预览检测效果
   └── python tools/preview_detection.py

4. 运行仿真
   └── python main.py

5. 查看回放结果
   └── 关闭窗口结束
```
