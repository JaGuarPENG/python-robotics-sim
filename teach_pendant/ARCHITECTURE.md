# 示教器模块架构文档

## 概述

`teach_pendant` 是一个基于 PyQt5 的机器人示教器 GUI 应用，用于连接和控制机器人。支持 3D 可视化、关节控制、跟随模式等功能。

## 目录结构

```
teach_pendant/
├── __init__.py          # 包初始化，导出主要类
├── __main__.py          # 支持 python -m teach_pendant 运行
├── config.py            # 配置参数
├── signals.py           # Qt 信号类定义
├── robot_3d_widget.py   # 3D 可视化控件
├── robot_controller.py  # 机器人控制器（核心业务逻辑）
├── main_window.py       # 主窗口 UI
└── app.py               # 应用程序入口

teach_pendant.py         # 根目录启动脚本
```

## 模块依赖关系

```
                    ┌─────────────────┐
                    │     app.py      │  应用入口
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  main_window.py │  主窗口
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│robot_3d_widget│   │robot_controller│   │   signals.py  │
│     .py       │   │     .py        │   │               │
└───────┬───────┘   └───────┬────────┘   └───────────────┘
        │                   │
        │           ┌───────▼────────┐
        │           │ FollowerClient │  (tools/follower_client.py)
        │           │   WebSocket    │
        │           └────────────────┘
┌───────▼───────┐
│  robot_dh.py  │  (tools/robot_dh.py)
│  create_ka_ur │
└───────────────┘
```

## 各模块详细说明

### 1. config.py - 配置参数

```python
# 连接配置
ROBOT_IP = "192.168.0.10"    # 机器人 IP
PORT_WS = 5999               # WebSocket 端口

# 预设位置 (关节角度，单位：度)
PRESET_POSITIONS = {
    "零位": [0, 0, 0, 0, 0, 0],
    "位置1": [0, -20, 90, 0, -45, 0],
    ...
}

# 速度参数
DEFAULT_VEL = [100, 200, 100]

# Jog 步进角度 (度)
JOG_STEP_SMALL = 1.0
JOG_STEP_MEDIUM = 5.0
JOG_STEP_LARGE = 10.0
```

### 2. signals.py - Qt 信号类

用于线程间安全通信的信号定义：

```python
class WorkerSignals(QObject):
    joints_updated = pyqtSignal(list)       # 关节位置更新 [j1, j2, j3, j4, j5, j6]
    status_updated = pyqtSignal(str)        # 状态消息文本
    connection_changed = pyqtSignal(bool, str)  # (是否连接, 端口类型)
    command_finished = pyqtSignal(bool, str)    # (是否成功, 消息)
    error_occurred = pyqtSignal(str)        # 错误消息
    robot_status_updated = pyqtSignal(dict) # 机器人状态字典
```

### 3. robot_3d_widget.py - 3D 可视化控件

基于 Matplotlib 的 3D 机械臂可视化：

```python
class Robot3DWidget(FigureCanvas):
    def __init__(self, parent=None)
    def setup_axes(self)           # 设置坐标轴
    def draw_base(self)            # 绘制底座
    def draw_origin_frame(self)    # 绘制坐标系
    def update_robot(self, joints_deg)  # 更新机器人位姿（关键方法）
    def get_joint_positions(self, joints_rad)  # 获取关节位置
    def reset_view(self)           # 重置视角
```

**关键属性：**
- `self.robot` - roboticstoolbox 机器人模型 (来自 tools/robot_dh.py)
- `self.current_joints` - 当前关节角度 (6个值)

### 4. robot_controller.py - 机器人控制器（核心）

封装所有与机器人的通信和控制逻辑：

```python
class RobotController:
    def __init__(self, signals: WorkerSignals)

    # 连接管理
    def connect(self, ip, port) -> bool
    def login(self, user, password) -> bool
    def logout(self) -> bool
    def enable(self) -> bool
    def stop(self)

    # 监控
    def start_monitoring(self)      # 启动关节位置监控线程
    def _monitor_loop(self)         # 监控循环 (20Hz)

    # 运动控制
    def move_joint(self, target_joints, vels)  # 关节运动
    def jog_joint(self, joint_index, direction, step)  # 单关节 Jog
    def set_velocity(self, percent)  # 设置速度百分比

    # 跟随模式
    def cmd_start_follower(self)    # start_follower 命令
    def cmd_set_jog_coordinate_tool(self)  # set_jog_coordinate --tool
    def cmd_follower_cart(self)     # follower_cart 命令
    def cmd_stop_follower(self)     # stop_follower 命令
    def start_follower_mode(self, ip)  # 一键启动跟随模式
    def stop_follower_mode(self)

    # 状态查询
    def get_robot_status(self) -> dict
    def _parse_status(self, data) -> dict
```

**关键属性：**
- `self.ws_client` - FollowerClientWebSocket 实例
- `self.is_logged_in` / `is_enabled` / `is_follower_mode` - 状态标志
- `self.current_joints` - 当前关节位置 (list)
- `self.actual_tcp` - TCP 位置 [x, y, z, rx, ry, rz]
- `self.data_lock` - 线程锁

### 5. main_window.py - 主窗口 UI

PyQt5 主窗口，包含所有 UI 组件和槽函数：

```python
class TeachPendantWindow(QMainWindow):
    def __init__(self)
    def init_ui(self)              # 初始化 UI（最大的方法）
    def connect_signals(self)      # 连接 Qt 信号
    def update_3d_view(self)       # 定时更新 3D 视图 (20Hz)

    # 槽函数
    def on_connect(self)           # 连接按钮
    def on_login(self) / on_logout(self) / on_enable(self)
    def on_cmd_start_follower(self)  # 跟随模式命令
    def on_cmd_set_jog_coordinate(self)
    def on_cmd_follower_cart(self)
    def on_cmd_stop_follower(self)
    def on_one_click_follower(self)  # 一键启动跟随

    # 状态更新
    def update_status(self, message)
    def update_joint_display(self, joints)
    def update_robot_status_display(self, status_info)

    # 控制
    def jog_joint(self, joint_index, direction)
    def move_to_preset(self, joints)
    def set_velocity(self, percent)
```

**UI 布局：**
```
┌─────────────────────────────────────────────────────────┐
│                    主窗口 (1200x800)                      │
├───────────────────────────┬─────────────────────────────┤
│                           │  连接控制                    │
│                           │  [IP] [连接] [登录] [使能]   │
│      3D 机械臂视图         ├─────────────────────────────┤
│      (Robot3DWidget)      │  跟随模式控制                │
│                           │  [start_follower] [...]     │
│                           ├─────────────────────────────┤
├───────────────────────────┤  关节位置 (J1-J6)           │
│  末端位置 (TCP)           ├─────────────────────────────┤
│  X Y Z Rx Ry Rz          │  关节控制 (Jog)             │
├───────────────────────────┤  [步进选择] [+/-按钮]       │
│  机器人状态               ├─────────────────────────────┤
│  运行/激活/运动/模式/错误  │  预设位置                   │
│                           ├─────────────────────────────┤
│                           │  速度控制                   │
└───────────────────────────┴─────────────────────────────┘
```

### 6. app.py - 应用入口

```python
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 设置深色主题调色板
    palette = QPalette()
    ...
    app.setPalette(palette)

    window = TeachPendantWindow()
    window.show()
    sys.exit(app.exec_())
```

## 数据流

### 关节位置更新流程

```
1. RobotController._monitor_loop()
   │  每 50ms 轮询一次
   ▼
2. ws_client.get_actual_joint_pos()
   │  从 WebSocket 获取关节位置
   ▼
3. signals.joints_updated.emit(joints)
   │  发送 Qt 信号
   ▼
4. TeachPendantWindow.update_joint_display()
   │  更新关节数值显示
   ▼
5. TeachPendantWindow.update_3d_view()  (定时器触发)
   │  更新 3D 可视化
   ▼
6. Robot3DWidget.update_robot(joints_deg)
      │  重绘机械臂
```

### 命令执行流程

```
1. UI 按钮点击 (如 on_enable)
   ▼
2. 创建后台线程执行命令
   threading.Thread(target=enable_task).start()
   ▼
3. RobotController.enable()
   ▼
4. ws_client.enable_robot()  (WebSocket 通信)
   ▼
5. signals.status_updated.emit("使能成功")
   ▼
6. TeachPendantWindow.update_status()  (更新 UI)
```

## 外部依赖

### tools/follower_client.py

```python
class FollowerClientWebSocket:
    def connect(self) -> bool
    def login(self, user, password) -> bool
    def logout(self) -> bool
    def enable_robot(self) -> bool
    def set_velocity(self, percent)
    def send_command(self, cmd, timeout) -> (bool, response)
    def get_actual_joint_pos(self) -> np.ndarray
    def init_follower_mode(self, login_first, skip_enable) -> bool
    def stop_follower_mode(self)
    def start_polling(self, interval)
    def close(self)
```

### tools/robot_dh.py

```python
def create_ka_ur() -> roboticstoolbox.Robot
    # 创建公司 UR 构型机器人模型
    # 用于 3D 可视化和正向运动学计算
```

## 扩展指南

### 添加新的预设位置

编辑 `config.py`:
```python
PRESET_POSITIONS = {
    "零位": [0, 0, 0, 0, 0, 0],
    "新位置": [10, -30, 60, 0, -45, 20],  # 添加新位置
    ...
}
```

### 添加新的控制命令

1. 在 `robot_controller.py` 添加方法:
```python
def cmd_new_command(self):
    success, _ = self.ws_client.send_command("new_command")
    if success:
        self.signals.status_updated.emit("命令执行成功")
    return success
```

2. 在 `main_window.py` 添加 UI 和槽函数:
```python
# init_ui 中
self.new_cmd_btn = QPushButton("新命令")
self.new_cmd_btn.clicked.connect(self.on_new_command)

# 槽函数
def on_new_command(self):
    def task():
        self.controller.cmd_new_command()
    threading.Thread(target=task, daemon=True).start()
```

### 修改 3D 可视化

编辑 `robot_3d_widget.py`:
- 修改 `setup_axes()` 调整坐标范围
- 修改 `link_colors` 调整连杆颜色
- 修改 `update_robot()` 调整绘制逻辑

## 运行方式

```bash
# 方式1：直接运行启动脚本
python teach_pendant.py

# 方式2：作为模块运行
python -m teach_pendant
```

## 注意事项

1. **线程安全**: 所有 UI 更新必须通过 Qt 信号，不能在后台线程直接操作 UI
2. **WebSocket 连接**: 使用单一 WebSocket 连接 (端口 5999)
3. **坐标单位**: 关节角度使用度 (deg)，TCP 位置使用毫米 (mm)
4. **跟随模式**: 跟随模式下禁用关节运动，需先停止跟随模式
