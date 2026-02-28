# KAANH_Digital_Twin 函数参考手册

本文档汇总 `KAANH_Digital_Twin` 模块下所有核心文件的类与函数说明，便于快速查阅。

---

## 目录

- [Core 核心模块](#core-核心模块)
  - [robot_state.py](#robot_statepy)
  - [safety_guard.py](#safety_guardpy)
- [Signals 信号系统](#signals-信号系统)
  - [signals.py](#signalspy)
- [Robot Controller 机器人控制器](#robot-controller-机器人控制器)
  - [robot_controller.py](#robot_controllerpy)
- [Logic 业务逻辑层](#logic-业务逻辑层)
  - [conveyor_tracking_service.py](#conveyor_tracking_servicepy)
  - [trajectory_service.py](#trajectory_servicepy)
- [Render 3D渲染层](#render-3d渲染层)
  - [robot_model.py](#robot_modelpy)
  - [robot_renderer.py](#robot_rendererpy)
- [UI 界面层](#ui-界面层)
  - [UI面板概览](#ui面板概览)

---

## Core 核心模块

### robot_state.py

**类：`RobotState`**

线程安全的机器人状态容器，集中管理所有实时状态数据。

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `is_connected` | bool | WebSocket 连接状态 |
| `is_logged_in` | bool | 登录状态 |
| `is_enabled` | bool | 机器人使能状态 |
| `is_follower_mode` | bool | 是否处于跟随模式 |
| `current_joints` | list[float] | 当前关节角度（6轴，单位：度） |
| `actual_tcp` | list[float] | 当前TCP位姿 [x,y,z,rx,ry,rz]（mm, deg） |
| `status_info` | dict | 运行状态字典 |
| `update_joints(joints)` | method | 更新关节角度（线程安全） |
| `get_joints()` | method | 获取关节角度副本 |
| `update_tcp(tcp)` | method | 更新TCP位姿（线程安全） |
| `get_tcp()` | method | 获取TCP位姿副本 |
| `update_status_info(info)` | method | 更新状态信息 |
| `get_status_info()` | method | 获取状态信息副本 |
| `joints_np` | property | 返回numpy格式的关节角（弧度） |

---

### safety_guard.py

**类：`SafetyGuard`**

运动安全卫士，负责运动指令的拦截与合法性检查。

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | `max_vel_ms=0.5`, `max_inc_mm=50.0` | - | 初始化安全卫士 |
| `check_increment(dx, dy, dz)` | 三轴位移增量 | `(bool, str)` | 检查单次增量是否超过安全阈值 |
| `check_velocity(dx_mm, dy_mm, dz_mm)` | 三轴位移（mm） | `(bool, str)` | 检查运动速度是否过快 |

---

## Signals 信号系统

### signals.py

**类：`WorkerSignals`**

Qt 信号类，继承自 `QObject`，用于线程间通信。

| 信号 | 参数类型 | 触发时机 |
|------|----------|----------|
| `joints_updated` | `list` | 关节位置更新 |
| `status_updated` | `str` | 状态消息通知 |
| `connection_changed` | `(bool, str)` | 连接状态变化（connected, port_type） |
| `command_finished` | `(bool, str)` | 命令执行完成（success, message） |
| `error_occurred` | `str` | 错误发生 |
| `robot_status_updated` | `dict` | 机器人详细状态更新 |
| `tracking_error_updated` | `(float, float)` | 跟踪误差更新（线性误差mm, 姿态误差deg） |
| `log_message` | `str` | 日志消息 |

**使用示例：**
```python
# 发射信号
self.signals.status_updated.emit("连接成功")

# 连接信号到槽函数（在主线程）
self.signals.status_updated.connect(self.on_status_updated)
```

---

## Robot Controller 机器人控制器

### robot_controller.py

**类：`RobotController`**

机器人控制器门面类，封装 WebSocket 和 UDP 通信，提供统一的控制接口。

#### 连接管理

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__(signals)` | `WorkerSignals` | - | 初始化控制器 |
| `connect(ip, port)` | `str`, `int` | `bool` | 建立WebSocket连接 |
| `login(user, password)` | `str`, `str` | `bool` | 登录系统 |
| `logout()` | - | `bool` | 退出登录 |
| `enable()` | - | `bool` | 使能机器人电机 |
| `start_follower_mode(ip)` | `str` | `bool` | 启动UDP跟随模式 |
| `cmd_stop_follower()` | - | - | 停止跟随模式 |
| `start_monitoring()` | - | - | 启动状态监控循环 |
| `stop_monitoring()` | - | - | 停止状态监控 |

#### 运动控制

| 方法 | 参数 | 说明 |
|------|------|------|
| `move_joint(target_joints, vels, wait_for_finish)` | `list`, `list`, `bool` | 关节运动 |
| `move_line(target_tcp, speed)` | `list`, `float` | 直线插补运动 |
| `send_udp_increment(dx, dy, dz, rx, ry, rz)` | 6个`float` | 发送UDP增量指令 |

#### 状态获取

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `get_actual_joints()` | `list` | 获取当前关节角度 |
| `get_actual_tcp()` | `list` | 获取当前TCP位姿 |
| `is_in_position(joints, tolerance)` | `bool` | 检查是否到达目标位置 |

---

## Logic 业务逻辑层

### conveyor_tracking_service.py

**类：`ConveyorTrackingService`**

传送带动态追踪服务，实现4阶段状态机：发现→悬停→逼近→复位。

| 属性/方法 | 说明 |
|-----------|------|
| `status_updated` | 信号，发送状态更新消息 |
| `loop_hz = 60.0` | 控制循环频率 |
| `state` | 当前状态：`OBSERVING`/`HOVERING`/`APPROACHING`/`RETURNING` |
| `tool_length = 0.20` | 探针长度（m） |
| `hover_height = 0.10` | 悬停高度（m） |
| `conveyor_speed_y = 0.05` | 传送带速度（m/s） |
| `xy_threshold = 0.010` | 水平误差阈值（m） |

| 方法 | 说明 |
|------|------|
| `set_conveyor_speed(speed)` | 动态修改追踪预测速度 |
| `start_tracking()` | 启动追踪服务 |
| `stop_tracking()` | 停止追踪服务 |
| `_tracking_loop()` | 主追踪循环（60Hz） |
| `_state_machine_logic(target_pos, target_id)` | 状态机逻辑处理 |
| `_send_command(pos, vel)` | 发送运动指令（带IK解算） |

---

### trajectory_service.py

**类：`TrajectoryService`**

轨迹执行服务，负责离线轨迹的IK解算与运动控制循环。

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__(controller, signals, fast_ik)` | 依赖注入 | - | 初始化服务 |
| `run_circular_trajectory(center, radius, num_points)` | `list`, `float`, `int` | `list` | 执行圆形轨迹（后台线程） |
| `move_to_target(target, current_joints)` | `list`, `list` | `(list, float)` | 移动到单个目标点 |

---

### vision_service.py

**类：`VisionService`**

视觉交互服务，负责与外部YOLO/相机流对接及坐标系变换。

| 方法 | 说明 |
|------|------|
| `start_camera()` | 启动相机采集 |
| `stop_camera()` | 停止相机采集 |
| `set_detection_mode(mode)` | 设置检测模式（YOLO/ArUco） |
| `get_detected_objects()` | 获取检测到的目标列表 |
| `pixel_to_world(u, v, depth)` | 像素坐标转世界坐标 |

---

## Render 3D渲染层

### robot_model.py

**类：`RobotModel`**

机器人几何模型与运动学计算封装。

| 属性/方法 | 说明 |
|-----------|------|
| `dh_robot` | DH参数模型（roboticstoolbox） |
| `urdf_robot` | URDF精细模型 |
| `link_colors` | 连杆颜色列表 |
| `stl_map` | STL文件映射关系 |
| `load_urdf()` | 加载URDF文件 |
| `get_joint_positions(joints_rad)` | 计算关节3D位置 |
| `get_ee_pose(joints_rad, use_urdf)` | 获取末端位姿 |

---

### robot_renderer.py

**类：`RobotRenderer`**

PyVista 3D渲染执行器，负责高精度模型渲染。

| 属性/方法 | 说明 |
|-----------|------|
| `plotter` | PyVista绘图器 |
| `model` | RobotModel实例 |
| `fine_actors` | STL模型Actor字典 |
| `trajectory_actors` | 轨迹线Actor列表 |
| `belt_objects` | 传送带物体Actor列表 |
| `setup_base_scene()` | 设置基础场景（网格、传送带） |
| `create_robot_actors()` | 加载STL创建Actor |
| `update_robot_pose(joints_rad)` | 更新机器人姿态 |
| `update_fov_pose(pose)` | 更新相机视野位置 |
| `add_trajectory_point(pos)` | 添加轨迹点 |
| `clear_trajectory()` | 清除轨迹显示 |
| `update_belt_objects(dt)` | 更新传送带物体位置 |
| `get_tracking_target()` | 获取当前追踪目标位置 |
| `toggle_fov_visibility(show)` | 切换视野显示 |

---

## UI 界面层

### UI面板概览

| 文件 | 类名 | 功能描述 |
|------|------|----------|
| `connection_panel.py` | `ConnectionPanel` | 连接面板：IP设置、登录/登出、使能控制 |
| `robot_status_panel.py` | `RobotStatusPanel` | 状态面板：关节角度、TCP坐标、运行状态显示 |
| `joint_control_panel.py` | `JointControlPanel` | 点动面板：关节/笛卡尔点动控制 |
| `teleop_panel.py` | `TeleopPanel` | 遥操作面板：手柄/键盘映射控制 |
| `follower_panel.py` | `FollowerPanel` | 随动面板：UDP跟随模式启停 |
| `vision_panel.py` | `VisionPanel` | 视觉面板：CSV轨迹加载、视觉追踪启停 |
| `log_panel.py` | `LogPanel` | 日志面板：实时显示系统日志 |

### 通用UI方法

所有UI面板通常包含以下方法：

| 方法 | 说明 |
|------|------|
| `__init__(controller, signals)` | 构造函数，注入控制器和信号 |
| `init_ui()` | 初始化UI布局 |
| `on_xxx_clicked()` | 按钮点击事件处理 |
| `update_status(data)` | 更新状态显示（槽函数） |

---

## 主窗口

### main_window.py

**类：`MainWindow`**

KAANH_Digital_Twin 主窗口，集成所有功能模块。

| 方法 | 说明 |
|------|------|
| `__init__()` | 初始化主窗口、控制器、信号连接 |
| `init_ui()` | 组装UI布局（左3D视图+右控制面板） |
| `init_tabs()` | 初始化标签页（基础/高级/视觉） |
| `connect_signals()` | 连接信号到槽函数 |
| `on_joints_updated(joints)` | 关节更新槽函数 |
| `on_connection_changed(connected, ptype)` | 连接状态变化槽函数 |
| `on_status_updated(msg)` | 状态消息槽函数 |
| `on_error_occurred(err)` | 错误处理槽函数 |
| `closeEvent(event)` | 窗口关闭事件，清理资源 |

---

## 快速索引

### 按功能查找

| 功能需求 | 相关类/文件 |
|----------|-------------|
| 获取机器人状态 | `RobotState` (core/robot_state.py) |
| 发送运动指令 | `RobotController` (robot_controller.py) |
| 线程安全通信 | `WorkerSignals` (signals.py) |
| 视觉追踪 | `ConveyorTrackingService` (logic/conveyor_tracking_service.py) |
| 轨迹执行 | `TrajectoryService` (logic/trajectory_service.py) |
| 3D显示更新 | `RobotRenderer` (render/robot_renderer.py) |
| 运动学计算 | `RobotModel` (render/robot_model.py) |
| 安全检查 | `SafetyGuard` (core/safety_guard.py) |

### 信号速查表

| 信号名 | 发射位置 | 接收位置（典型） |
|--------|----------|------------------|
| `joints_updated` | RobotController._monitor_loop | MainWindow.on_joints_updated → Robot3DWidget |
| `status_updated` | 各Service | MainWindow, LogPanel |
| `connection_changed` | RobotController | ConnectionPanel, MainWindow |
| `tracking_error_updated` | TrajectoryService | VisionPanel |
| `log_message` | StreamRedirector | LogPanel |

---

*文档版本: v1.0.0 | 更新日期: 2026-02-28*
