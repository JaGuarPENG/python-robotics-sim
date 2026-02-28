# 开发指南

## 1. 环境搭建

### 1.1 必需工具

| 工具 | 版本 | 用途 |
|------|------|------|
| Python | >=3.8 | 运行时 |
| Git | >=2.30 | 版本控制 |
| Visual Studio C++ Build Tools | - | 编译 C++ 扩展 (如 FastIK 需要重新编译) |

### 1.2 本地开发环境配置

```bash
# 1. 克隆代码库
git clone https://your-repo-address/python-robotics-sim.git
cd python-robotics-sim

# 2. 创建并激活虚拟环境 (可选)
conda create -n robosim python=3.9
conda activate robosim

# 3. 安装依赖包
pip install -r requirements.txt
# 或者手动安装核心依赖：
pip install PyQt5 pyvista numpy pandas scipy

# 4. 启动开发服务器（主界面）
python -m teach_pendant.app  # 启动 KAANH_Digital_Twin
```

## 2. 项目结构说明

### 2.1 目录结构

```
teach_pendant/  # KAANH_Digital_Twin 源码目录
├── app.py                 # 应用入口
├── main_window.py         # 主窗口
├── robot_controller.py    # 机器人控制器
├── robot_3d_widget.py     # 3D 渲染控件
├── signals.py             # 信号总线
├── config.py              # 配置文件
├── core/                  # 核心模块
│   ├── robot_state.py     # 机器人状态管理
│   └── safety_guard.py    # 安全保护
├── logic/                 # 业务逻辑层
│   ├── conveyor_tracking_service.py
│   ├── trajectory_service.py
│   └── vision_service.py
├── render/                # 3D 渲染
│   ├── robot_model.py
│   └── robot_renderer.py
└── ui/                    # UI 面板
    ├── connection_panel.py
    ├── robot_status_panel.py
    ├── joint_control_panel.py
    ├── teleop_panel.py
    ├── follower_panel.py
    ├── vision_panel.py
    └── log_panel.py
```

### 2.2 代码分层

- **UI 层** (`ui/`): 处理用户交互，严禁执行耗时操作
- **Logic 层** (`logic/`): 承载业务逻辑，运行在独立线程
- **Core 层** (`core/`): 状态管理和安全防护
- **Render 层** (`render/`): 3D 渲染相关

## 3. 代码规范

### 3.1 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件 | snake_case | `robot_controller.py` |
| 类 | PascalCase | `RobotController` |
| 方法 | snake_case | `move_joint` |
| 常量 | UPPER_SNAKE_CASE | `DEFAULT_VEL` |
| 私有属性 | _前缀 | `_position_lock` |

### 3.2 UI 与逻辑强制解耦

KAANH_Digital_Twin 最基础的开发准则：**严禁在 UI (Main Thread) 中执行耗时（> 16ms）的任务。**

所有涉及网络 I/O (`time.sleep()`, `socket.recv()`) 或密集计算的动作，必须交由 `logic` 层中的服务在独立的 `threading.Thread` 中处理。

### 3.3 多线程安全 (Signal-Driven)

由于后台线程无法直接修改 PyQt 控件（如 `QLabel.setText()`），我们利用 `teach_pendant/signals.py` 作为 KAANH_Digital_Twin 的统一事件总线：

**错误示范（将导致崩溃）**：
```python
# 在子线程内直接操作 UI
def background_task():
    self.label_status.setText("连接成功")  # ❌ 致命错误
```

**正确示范（使用 Signal）**：
```python
def background_task():
    # 后台只负责发信号，主线程的槽函数会安全地更新 UI
    self.signals.status_updated.emit("连接成功")  # ✅ 正确做法
```

### 3.4 注释规范

```python
class RobotController:
    """
    机器人控制器 - 负责底层通信与指令分发
    
    封装了 WebSocket 和 UDP 两种通信方式，提供统一的运动控制接口。
    内部包含 RobotState 实例用于线程安全的状态管理。
    
    Attributes:
        signals: WorkerSignals 实例，用于线程间通信
        state: RobotState 实例，机器人当前状态
        safety: SafetyGuard 实例，安全保护
    """
    
    def move_joint(self, target_joints, vels=None, wait_for_finish=True):
        """
        控制机器人关节运动
        
        Args:
            target_joints: 目标关节角度数组（6个元素，单位：度）
            vels: 各关节速度，默认 [30,30,30,30,30,30]
            wait_for_finish: 是否等待运动完成，默认 True
            
        Returns:
            bool: 命令发送是否成功
            
        Raises:
            RuntimeError: 当机器人未使能时
        """
```

## 4. 常见开发任务教程

### 4.1 如何新增一个 UI 控制面板？

假设你需要开发一个控制外部夹爪的 `GripperPanel`。

**步骤 1：在 `teach_pendant/ui/` (KAANH_Digital_Twin UI 目录) 下创建 `gripper_panel.py`**

```python
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from ..robot_controller import RobotController

class GripperPanel(QWidget):
    def __init__(self, controller: RobotController, signals):
        super().__init__()
        self.controller = controller
        self.signals = signals
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.btn_open = QPushButton("打开夹爪")
        self.btn_close = QPushButton("闭合夹爪")
        
        self.btn_open.clicked.connect(self.on_open_clicked)
        self.btn_close.clicked.connect(self.on_close_clicked)
        
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_close)

    def on_open_clicked(self):
        # 夹爪动作可能是非阻塞的或交由后台发指令
        self.signals.status_updated.emit("正在打开夹爪...")
        # self.controller.cmd_open_gripper() # 假设你扩充了相关 API
```

**步骤 2：将新面板注册到主窗口**

打开 `teach_pendant/main_window.py`，在 `init_ui` 方法的 Tab 页组装部分：

```python
from .ui.gripper_panel import GripperPanel

# ... 在对应的 Tab 下面添加
self.gripper_panel = GripperPanel(self.controller, self.signals)
advanced_layout.addWidget(self.gripper_panel)
```

### 4.2 如何扩展 `RobotController` 通信指令？

如果底层控制器新增了特定的 JSON-RPC 指令（例如修改加速度）。

打开 `teach_pendant/robot_controller.py`，增加封装函数：

```python
def set_global_acceleration(self, acc_percent: int):
    """
    修改机器人的全局加速度比例
    
    Args:
        acc_percent: 加速度百分比 (1-100)
        
    Returns:
        bool: 设置是否成功
    """
    if not self.ws_connected:
        return False
        
    payload = {
        "method": "set_sys_acc",
        "params": [acc_percent],
        "id": 1001
    }
    
    # 假设底层的 ws_client 提供了 send_json 方法
    success = self.ws_client.send_json(payload)
    if success:
        self.signals.status_updated.emit(f"全局加速度已设置为 {acc_percent}%")
    return success
```

## 5. 调试技巧

### 5.1 UI 假死问题排查

如果在执行某项功能时，程序界面无法拖动，通常是因为某处违背了异步原则。

**排查路径**：
1. 检查对应的按钮 `clicked.connect()` 槽函数，里面是否含有 `while True:` 循环？
2. 是否直接调用了 `controller.move_joint(..., wait_for_finish=True)`？

如果是，请将其用 `threading.Thread(target=..., daemon=True).start()` 包裹起来。

### 5.2 3D 渲染闪退 (Segmentation Fault)

- 确保在主线程中只通过定时器 (`QTimer`) 或事件机制来刷新 `Robot3DWidget`
- 如果加载庞大的 STL 模型或高频刷新轨迹点，确保 `pyvista` 渲染器的 `update()` 在安全范围内调用，避免与其他视图对象的更新发生抢占冲突

### 5.3 查看底层日志输出

本系统拦截了标准输出 `sys.stdout`，将其重定向到了前端底部的 `LogPanel` 控件。如果发现某些底层 C++ 错误日志无法在面板显示，可临时注释掉 `main_window.py` 中的 `self._setup_log_redirection()` 方法，让系统恢复向终端控制台打印，以方便阅读详细 Traceback。

## 6. 提交规范

### 6.1 Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 6.2 Type 说明

| 类型 | 说明 |
|------|------|
| feat | 新功能 |
| fix | 修复 bug |
| docs | 文档更新 |
| style | 代码格式调整 |
| refactor | 重构 |
| test | 测试相关 |
| chore | 构建/工具相关 |

### 6.3 示例

```
feat(ui): 添加夹爪控制面板

实现夹爪的打开/闭合控制功能
- 添加 GripperPanel 类
- 集成到主窗口的 Advanced Tab
- 添加相关信号处理

Closes #123
```

## 7. 代码审查清单

- [ ] 代码符合项目规范
- [ ] UI 操作都在主线程，耗时操作都在后台线程
- [ ] 使用了 Signal 进行线程间通信
- [ ] 添加了必要的注释
- [ ] 没有引入安全漏洞
- [ ] 错误处理完善
