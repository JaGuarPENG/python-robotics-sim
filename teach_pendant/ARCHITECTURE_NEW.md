# 示教器模块最终架构文档 (Final Architecture)

## 1. 系统概览
本模块重构后采用了 **MVC-S (Model-View-Controller-Service)** 的演进架构，实现了界面、数据、逻辑与通信的四维分离。

## 2. 详细目录结构
```text
teach_pendant/
├── ui/                 # 【View】表现层组件
│   ├── connection_panel.py   # 连接/登录控制
│   ├── robot_status_panel.py # 状态/TCP 显示
│   ├── joint_control_panel.py # 关节 Jog/速度调节
│   ├── follower_panel.py     # 跟随模式指令
│   └── teleop_panel.py       # UDP 遥操作测试
├── core/               # 【Model】数据核心层
│   ├── robot_state.py      # 线程安全的状态容器
│   └── safety_guard.py     # 运动安全校验哨兵
├── logic/              # 【Service】业务逻辑层
│   └── trajectory_service.py # 轨迹生成与 IK 解算循环
├── render/             # 【Render】3D 渲染执行层
│   ├── robot_model.py      # 渲染专用几何数据与 FK
│   └── robot_renderer.py   # PyVista 场景渲染器
├── main_window.py      # 【Mediator】主窗口组装与路由
├── robot_controller.py # 【Facade】机器人通信统一入口
├── robot_3d_widget.py  # 【Qt Widget】3D 渲染控件包装器
├── signals.py          # 全局信号总线
├── config.py           # 静态配置参数
└── app.py              # 应用入口
```

## 3. 核心设计模式

### 3.1 状态中台模式 (RobotState)
所有实时数据（关节、TCP、状态位）不再散落在 Controller 各处，而是统一存储在 `RobotState` 中。Controller 负责通过 `update_*` 方法写入，UI 及其他组件通过 `get_*` 方法读取，确保了单一数据源。

### 3.2 策略安全拦截 (SafetyGuard)
所有的 UDP 运动指令在发出前必须通过 `SafetyGuard` 的双重校验（位移增量限值 + 瞬时速度限值），从架构层面杜绝了危险指令的下发。

### 3.3 逻辑外包服务 (TrajectoryService)
耗时的轨迹点生成与 IK 递归解算被封装为独立服务，运行在独立的后台线程中，通过信号与主窗口通信，确保了 UI 界面的丝滑响应。

### 3.4 渲染管线分离 (RobotRenderer)
3D 渲染不再由 Qt Widget 直接操作几何点，而是由 `RobotRenderer` 管理 PyVista 的 Actor 生命周期。这种设计允许在不改动 UI 逻辑的前提下，通过优化 Renderer 内部实现来提升帧率。

## 4. 协作流程 (典型场景：Jog 运动)
1.  **UI 触发**：`JointControlPanel` 捕获按钮点击，调用 `Controller.jog_joint()`。
2.  **数据读取**：`Controller` 从 `RobotState` 获取当前关节。
3.  **指令构建**：`Controller` 构建 JSON 指令字符串。
4.  **通信发送**：通过 `FollowerClientWebSocket` 将指令送往物理控制器。
5.  **反馈更新**：后台监控线程收到最新关节数据，写入 `RobotState` 并发射 `signals.joints_updated`。
6.  **渲染重绘**：`main_window` 响应信号，驱动 `robot_3d_widget` 调用其内部的 `renderer.update()`。

---
*本文档反映了 2026-02-09 重构后的最新生产环境架构。*