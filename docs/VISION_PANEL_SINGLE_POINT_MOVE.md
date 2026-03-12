# 视觉引导 - 单点遥操作移动功能说明

## 功能概述

在 KAANH_Digital_Twin 的"视觉引导"选项卡中，新增了**单点遥操作移动**功能。用户可以输入目标坐标（XYZRxRyRz），机械臂会自动通过 UDP 增量模式移动到该位置。

## 界面位置

位于 **"视觉引导"** 选项卡的最下方，标题为 **"单点遥操作移动 (UDP增量模式)"**。

## 使用步骤

### 1. 获取当前坐标
点击 **"🔄 获取当前坐标"** 按钮：
- 系统会读取机械臂当前位置
- 当前坐标显示在按钮上方的绿色标签中
- 目标坐标输入框会自动填充当前值（方便微调）

### 2. 输入目标坐标
在输入框中修改目标坐标（单位：mm/°）：
- **X (mm)**: 目标 X 坐标
- **Y (mm)**: 目标 Y 坐标
- **Z (mm)**: 目标 Z 坐标
- **Rx (°)**: 目标 Rx 角度
- **Ry (°)**: 目标 Ry 角度
- **Rz (°)**: 目标 Rz 角度

💡 **提示**: 输入框已自动填充当前坐标，只需修改需要调整的数值即可。

### 3. 执行移动
点击 **"▶ 执行移动到目标坐标"** 按钮：
- 系统会弹出确认对话框，显示：
  - 当前位置
  - 目标位置
  - 偏移量
- 点击 **"是"** 开始移动
- 机械臂会自动：
  1. 连接 UDP（如未连接）
  2. 启动跟随模式（如未启动）
  3. 通过 UDP 增量模式移动到目标位置
  4. 持续 5 秒确保到位

## 坐标映射说明

根据测试验证，UDP 发送的坐标映射关系如下：

| 基座坐标偏移 | UDP 发送值 | 映射关系 |
|-------------|-----------|---------|
| offset_x (X方向) | `send_x = offset_x / 1000` | 直接对应 |
| offset_y (Y方向) | `send_y = -offset_y / 1000` | **取反** |
| offset_z (Z方向) | `send_z = -offset_z / 1000` | **取反** |

### 示例
从当前位置 (538.4, 357.2, 713.8) 移动到 (460.0, 278.8, 713.8)：
```
偏移量: X=-78.4mm, Y=-78.4mm, Z=0mm

UDP发送:
  send_x = -78.4/1000 = -0.0784  (直接)
  send_y = -(-78.4)/1000 = +0.0784  (取反)
  send_z = 0
```

## 注意事项

1. **机器人必须使能**: 使用前确保机器人已连接并使能
2. **确保工作空间安全**: 移动前确认路径无障碍物
3. **移动持续时间**: 每次移动持续 5 秒，期间会连续发送 UDP 增量指令
4. **频率**: 发送频率为 50Hz (每 20ms 发送一次)
5. **角度修正**: 工控机返回的角度顺序为 [rz, ry, rx]，系统已自动修正为 [rx, ry, rz]

## 技术实现

### 相关文件
- `KAANH_Digital_Twin/ui/vision_panel.py`: UI 界面实现
- `KAANH_Digital_Twin/main_window.py`: 信号处理与移动逻辑
- `tools/follower_client.py`: UDP 通信与角度修正

### 信号流程
```
用户点击"获取当前坐标"
  ↓
VisionPanel.get_current_position_requested.emit()
  ↓
MainWindow._on_get_current_position_requested()
  ↓
VisionPanel.update_current_position_display()

用户点击"执行移动"
  ↓
VisionPanel.on_single_point_move_clicked()
  ↓
VisionPanel.single_point_move_requested.emit(x, y, z, rx, ry, rz)
  ↓
MainWindow._on_single_point_move_requested()
  ↓
MainWindow._execute_single_point_move() (后台线程)
  ↓
Controller.send_pose_euler(send_x, send_y, send_z, dry, drx, drz)
```

## 故障排除

### 无法获取当前坐标
- 检查机器人是否已连接并使能
- 检查 WebSocket 连接状态

### 移动方向不正确
- 参考坐标映射说明，确认 X/Y/Z 方向的映射关系
- 如有问题，运行 `test_udp_follower.py` 选择选项 3 重新测试坐标映射

### 移动不到位
- 检查 UDP 连接是否正常
- 检查跟随模式是否已启动
- 查看日志输出是否有错误信息

## 相关文档
- [UDP 遥操作坐标映射规范](./UDP_TELEOPERATION_COORDINATE_MAPPING.md)
