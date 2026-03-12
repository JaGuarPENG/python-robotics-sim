# UDP 遥操作坐标映射规范

## 概述

本文档详细说明在使用 UDP 增量模式（follower_cart）进行机器人遥操作时，基座坐标系与工具坐标系之间的映射关系。

## 1. 坐标系定义

### 1.1 基座坐标系（Base Frame）
- **X 轴**: 机器人前方（正方向）
- **Y 轴**: 机器人左侧（正方向）
- **Z 轴**: 垂直向上（正方向）
- **Rx**: 绕 X 轴旋转（Roll）
- **Ry**: 绕 Y 轴旋转（Pitch）
- **Rz**: 绕 Z 轴旋转（Yaw）

### 1.2 工具坐标系（Tool Frame）
 follower_cart 模式下，UDP 发送的增量是相对于工具坐标系的：
- **tool_x**: 工具 X 方向偏移
- **tool_y**: 工具 Y 方向偏移
- **tool_z**: 工具 Z 方向偏移

## 2. 角度映射（关键修复）

### 2.1 工控机返回角度顺序
工控机通过 WebSocket 返回的旋转角度顺序为 `[rz, ry, rx]`，**不是**标准的 `[rx, ry, rz]`。

```python
# 原始数据 (pe[0] 来自工控机)
# pe[0][0:3] = [x, y, z]
# pe[0][3] = rz
# pe[0][4] = ry
# pe[0][5] = rx

# 转换为标准顺序 [rx, ry, rz]
tcp_pe_corrected = [
    pe_raw[0], pe_raw[1], pe_raw[2],  # x, y, z
    pe_raw[5], pe_raw[4], pe_raw[3]   # rx, ry, rz (交换顺序)
]
```

### 2.2 UDP 发送角度映射
UDP 数据包中的角度参数顺序：
- `param[3]` = dry (对应 Ry)
- `param[4]` = drx (对应 Rx)
- `param[5]` = drz (对应 Rz)

```python
# 旋转映射
send_pose(x, y, z, dry, drx, drz)
# 其中:
# dry = np.deg2rad(offset_ry)
# drx = np.deg2rad(offset_rx)
# drz = np.deg2rad(offset_rz)
```

## 3. 位置映射（核心规律）

### 3.1 映射公式

| 基座坐标偏移 | UDP 发送值 | 映射关系 | 说明 |
|-------------|-----------|---------|------|
| `offset_x` (基座X) | `send_x = offset_x / 1000` | **直接对应** | X 方向直接发送 |
| `offset_y` (基座Y) | `send_y = -offset_y / 1000` | **取反** | Y 方向需要取反 |
| `offset_z` (基座Z) | `send_z = -offset_z / 1000` | **取反** | Z 方向需要取反 |

### 3.2 映射原理

```python
# 目标位置 - 当前位置 = 偏移量 (mm)
offset_x = target_x - current_x
offset_y = target_y - current_y
offset_z = target_z - current_z

# 转换为 UDP 发送值 (m)
send_x = offset_x / 1000.0          # X: 直接对应
send_y = -offset_y / 1000.0         # Y: 取反
send_z = -offset_z / 1000.0         # Z: 取反

# 发送 UDP 增量
send_pose(send_x, send_y, send_z, dry, drx, drz)
```

## 4. 验证示例

### 示例 1: X 方向移动
```
当前位置: X=538.4 mm
目标位置: X=460.0 mm
偏移量:   offset_x = -78.4 mm

UDP 发送值:
  send_x = -78.4 / 1000 = -0.0784 m
  send_y = 0
  send_z = 0

调用: send_pose(-0.0784, 0, 0, 0, 0, 0)
结果: 机器人 X 方向移动 -78.4 mm (向负方向) ✅
```

### 示例 2: Y 方向移动
```
当前位置: Y=357.2 mm
目标位置: Y=278.8 mm
偏移量:   offset_y = -78.4 mm

UDP 发送值:
  send_x = 0
  send_y = -(-78.4) / 1000 = +0.0784 m  (注意取反!)
  send_z = 0

调用: send_pose(0, 0.0784, 0, 0, 0, 0)
结果: 机器人 Y 方向移动 -78.4 mm (向负方向) ✅
```

### 示例 3: 综合移动
```
当前位置: X=538.4, Y=357.2, Z=713.8 mm
目标位置: X=460.0, Y=278.8, Z=700.0 mm

偏移量:
  offset_x = -78.4 mm
  offset_y = -78.4 mm
  offset_z = -13.8 mm

UDP 发送值:
  send_x = -0.0784 m      (X 直接)
  send_y = +0.0784 m      (Y 取反: -(-78.4)/1000)
  send_z = +0.0138 m      (Z 取反: -(-13.8)/1000)

调用: send_pose(-0.0784, 0.0784, 0.0138, dry, drx, drz)
```

## 5. 完整代码实现

### 5.1 位置修正（工控机数据解析）
```python
# tools/follower_client.py
pe_raw = pe[0][:6]
self._actual_pe = np.array([
    pe_raw[0], pe_raw[1], pe_raw[2],  # x, y, z
    pe_raw[5], pe_raw[4], pe_raw[3]   # rx, ry, rz (交换顺序)
], dtype=np.float64)
```

### 5.2 单点遥操作移动
```python
def move_to_target(current_pos, target_pos):
    """从当前位置移动到目标位置"""
    # 1. 计算偏移量 (mm)
    offset_x = target_pos[0] - current_pos[0]
    offset_y = target_pos[1] - current_pos[1]
    offset_z = target_pos[2] - current_pos[2]
    offset_rx = target_pos[3] - current_pos[3]
    offset_ry = target_pos[4] - current_pos[4]
    offset_rz = target_pos[5] - current_pos[5]
    
    # 2. 转换为 UDP 发送值 (m, rad)
    send_x = offset_x / 1000.0          # X 直接对应
    send_y = -offset_y / 1000.0         # Y 取反
    send_z = -offset_z / 1000.0         # Z 取反
    
    dry = np.deg2rad(offset_ry)
    drx = np.deg2rad(offset_rx)
    drz = np.deg2rad(offset_rz)
    
    # 3. 持续发送增量 (50Hz，持续 5 秒)
    for _ in range(250):  # 5s * 50Hz
        send_pose(send_x, send_y, send_z, dry, drx, drz)
        time.sleep(0.02)
```

## 6. 常见问题

### Q1: 为什么 Y 和 Z 需要取反而 X 不需要？
这与 follower_cart 模式的内部坐标系定义有关。工具坐标系与基座坐标系在 Y 和 Z 轴方向相反。

### Q2: 旋转角度是否需要取反？
不需要。Rx、Ry、Rz 按照映射关系直接发送即可：
- `dry` = Ry (index 3)
- `drx` = Rx (index 4)  
- `drz` = Rz (index 5)

### Q3: 如何验证坐标映射是否正确？
使用 `test_udp_follower.py` 中的选项 3 (坐标映射测试)，依次测试 X+、X-、Y+、Y- 四个方向，观察机器人实际移动方向。

## 7. 相关文件

- `tools/follower_client.py`: UDP 客户端，包含角度顺序修正
- `test_udp_follower.py`: 测试脚本，包含坐标映射测试
- `KAANH_Digital_Twin/main_window.py`: UI 主窗口，包含单点遥操作实现
- `KAANH_Digital_Twin/ui/vision_panel.py`: 视觉面板 UI

## 8. 版本历史

| 版本 | 日期 | 修改内容 |
|-----|------|---------|
| v1.0 | 2026-03-12 | 初始文档，总结坐标映射规律 |
