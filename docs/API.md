# API 接口文档

## 1. 接口概览

本文档定义了 `KAANH_Digital_Twin` 内部核心模块（主要指 `RobotController` 和 `WorkerSignals`）的方法级接口。这些 API 并非通过 HTTP 暴露的网络接口，而是供系统内部其他模块（如 UI 面板或逻辑服务层）调用的 Python 方法签名与信号协议。

> **说明**：KAANH_Digital_Twin 向下与实体机器人交互使用的是私有 WebSocket 和 UDP 协议，这些协议的细节已被 `RobotController` 屏蔽。开发者在扩展新功能时，仅需调用本文档列出的 API。

## 2. 信号系统 (WorkerSignals)

`WorkerSignals` 是系统跨线程通信的神经中枢。所有继承自 `QThread` 或 `threading.Thread` 的后台服务必须通过触发这些信号来更新前端界面。

### 2.1 状态与日志更新

#### `status_updated`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(str)` |
| 用途 | 用于在底部状态栏或 Log 面板打印常规的系统信息 |

**参数**：
- `message` (str): 要显示的文本内容

**示例**：
```python
self.signals.status_updated.emit("步骤 1/3: 正在移动到轨迹起点...")
```

#### `log_message`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(str)` |
| 用途 | 专门用于接收重定向后的 `sys.stdout` 输出流 |

**参数**：
- `message` (str): 打印出的控制台文本

### 2.2 连接状态更新

#### `connection_changed`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(bool, str)` |
| 用途 | 当 WebSocket 或 UDP 的连接状态发生变化时触发，UI 根据此信号更新按钮颜色或文本 |

**参数**：
- `connected` (bool): `True` 表示已连接，`False` 表示断开
- `protocol_type` (str): `"WebSocket"` 或 `"UDP"`

**示例**：
```python
self.signals.connection_changed.emit(True, "WebSocket")
```

### 2.3 数据指标监控

#### `tracking_error_updated`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(float, float)` |
| 用途 | 在执行轨迹跟随任务时，输出实际 TCP 与目标规划点之间的实时物理偏差 |

**参数**：
- `linear_error` (float): 直线距离偏差（单位：毫米）
- `angular_error` (float): 姿态角度偏差（预留）

### 2.4 关节与状态更新

#### `joints_updated`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(list)` |
| 用途 | 关节位置更新 |

**参数**：
- `joints` (list[float]): 6个关节角度（单位：度）

#### `robot_status_updated`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(dict)` |
| 用途 | 机器人详细状态更新 |

**参数**：
- `status` (dict): 包含机器人状态信息的字典

#### `command_finished`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(bool, str)` |
| 用途 | 命令执行完成通知 |

**参数**：
- `success` (bool): 是否成功
- `message` (str): 完成消息

#### `error_occurred`

| 属性 | 值 |
|------|-----|
| 类型 | `pyqtSignal(str)` |
| 用途 | 错误发生通知 |

**参数**：
- `error` (str): 错误信息

## 3. 硬件抽象层 (RobotController)

`RobotController` 管理着机器人的所有连接会话及指令下发，内部包含用于数据安全存储的 `RobotState` 实例。

### 3.1 连接与初始化

#### `connect(ip: str, port: int) -> bool`

| 属性 | 值 |
|------|-----|
| 描述 | 通过给定的 IP 地址和端口，与机器人控制器建立 WebSocket 连接。连接成功后会自动启动后台监听线程以接收机器人的高频状态上报 |

**参数**：
- `ip` (str): 控制器的 IPv4 地址
- `port` (int): WebSocket 端口号

**返回值**：
- `bool`: 成功返回 `True`，否则返回 `False` 并将异常发送给 `signals`

#### `login(user: str, password: str) -> bool`

| 属性 | 值 |
|------|-----|
| 描述 | 登录机器人控制系统 |

**参数**：
- `user` (str): 用户名，默认 `"Engineer"`
- `password` (str): 密码，默认 `"000000"`

**返回值**：
- `bool`: 登录成功返回 `True`

#### `logout() -> bool`

| 属性 | 值 |
|------|-----|
| 描述 | 退出登录 |

**返回值**：
- `bool`: 退出成功返回 `True`

#### `enable() -> bool`

| 属性 | 值 |
|------|-----|
| 描述 | 使能机器人电机 |

**返回值**：
- `bool`: 使能成功返回 `True`

#### `start_follower_mode(ip: str) -> bool`

| 属性 | 值 |
|------|-----|
| 描述 | 一键式开启笛卡尔高速跟随模式 (`follower_cart`)。底层会自动发指令切换运动模式，并唤起 UDP Client，等待接收外部连续高频的位置下发 |

**参数**：
- `ip` (str): 控制器的 UDP 端口 IP 地址

**返回值**：
- `bool`: 模式进入成功返回 `True`

#### `cmd_stop_follower()`

| 属性 | 值 |
|------|-----|
| 描述 | 退出跟随模式，恢复到基础的点动待机状态 (`standby`)，同时切断 UDP 连接流 |

### 3.2 基础运动控制 (基于 WebSocket)

#### `move_joint(target_joints: list, vels: list = None, wait_for_finish: bool = True)`

| 属性 | 值 |
|------|-----|
| 描述 | 控制机器人的 6 个关节移动到指定角度 |

**参数**：
- `target_joints` (list[float]): 目标关节角度数组（单位：度）
- `vels` (list[float]): 各关节的百分比速度或绝对速度，默认 `[30,30,30,30,30,30]`
- `wait_for_finish` (bool): 若为 `True`，则阻塞当前线程直到运动到位；若为 `False`，则以非阻塞异步方式下发指令（适用于连续小线段发送）

> **注意**：不应在 UI 主线程中传递 `wait_for_finish=True` 进行调用

#### `move_line(target_tcp: list, speed: float = 50.0)`

| 属性 | 值 |
|------|-----|
| 描述 | 执行笛卡尔空间的直线插补运动 |

**参数**：
- `target_tcp` (list[float]): 目标 TCP 位姿 `[x, y, z, rx, ry, rz]`（单位：mm, deg）
- `speed` (float): 直线运行速度（单位：mm/s）

### 3.3 高速位姿下发 (基于 UDP)

#### `udp_client.send_pose_euler(dx, dy, dz, rx, ry, rz)`

| 属性 | 值 |
|------|-----|
| 描述 | 在 `follower_cart` 模式下，通过无连接的 UDP 端口向底层发送微小的位姿**增量**补偿 |

**参数**：
- `dx, dy, dz` (float): 三维空间的位置偏移量（单位：**米** m）
- `rx, ry, rz` (float): 三轴的欧拉角旋转偏移量（单位：**弧度** rad）

**要求频率**：需在外部线程以 >= 30Hz 的速率周期性调用，以防驱动器进入看门狗超时保护

## 4. 逆解计算引擎 (FastIKSolver)

此模块利用 C++ (或 Python 扩展) 的性能优势提供低延迟的数学解算。

### 4.1 方法

#### `solve_ik(pos: list, rpy_deg: list, current_joints: list) -> tuple`

| 属性 | 值 |
|------|-----|
| 描述 | 计算指定笛卡尔位姿对应的最优（距离当前角度最近）关节角组合 |

**参数**：
- `pos` (list[float]): 目标末端位置 `[x, y, z]` (单位：**米** m)
- `rpy_deg` (list[float]): 目标末端欧拉角 `[rx, ry, rz]` (单位：**度** deg)
- `current_joints` (list[float]): 当前各关节的真实角度 `[q1...q6]` (单位：**弧度** rad)，作为数值求解的初始种子点

**返回值**：
- `(target_joints_rad, ik_flags)`: `target_joints_rad` 为求解出的关节角（弧度制）列表，若无解或奇异则返回 `None`。`ik_flags` 携带诊断信息
