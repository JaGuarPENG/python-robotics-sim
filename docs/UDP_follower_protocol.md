# UDP Follower 遥操作通信协议文档

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    KAANH_Digital_Twin                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ main_window   │    │ robot_       │    │ follower_    │   │
│  │ (轨迹调度)    │───>│ controller   │───>│ client       │   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘   │
│                                                  │           │
└──────────────────────────────────────────────────┼───────────┘
                                                   │
                        ┌──────────────────────────┼────────┐
                        │                          │        │
                        │    ┌─────────────────────┼──────┐ │
                        │    │  WebSocket 5888     UDP 9998│ │
                        │    │  (控制指令/状态)   (位姿数据)│ │
                        │    └─────────────────────┼──────┘ │
                        │         工控机 192.168.0.10       │
                        └───────────────────────────────────┘
```

### 两条通信通道

| 通道 | 协议 | 端口 | 方向 | 用途 |
|------|------|------|------|------|
| WebSocket | WS (Binary) | 5888 | 双向 | 控制指令 (登录/使能/跟随模式启停) + 状态轮询 |
| UDP | UDP (Datagram) | 9998 | 单向 (发送) | 发送位姿目标 (follower_cart 模式下) |

> **注意**: UDP 9998 端口是单向的，工控机不回包。机器人实时位姿只能通过 WebSocket 轮询获取。


## 2. 通信协议格式

### 2.1 ARIS 消息头 (40 字节，所有消息通用)

```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ msg_len  │ msg_id   │ msg_type │reserved1 │reserved2 │reserved3 │
│  4 bytes │  4 bytes │  8 bytes │  8 bytes │  8 bytes │  8 bytes │
│  uint32  │  uint32  │  uint64  │  int64   │  int64   │  int64   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

Python 打包：
```python
HEADER_FORMAT = '<IIQqqq'   # 小端序, 共 40 字节
header = struct.pack(HEADER_FORMAT, len(payload), 0, 0, 0, 0, 0)
packet = header + payload   # 完整数据包
```


### 2.2 UDP 位姿数据包 (发送给工控机)

header (40B) + JSON payload：
```json
{
    "type": "321",
    "pe": [[x, y, z, rx, ry, rz]],
    "pq": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| type | string | 欧拉角类型，固定 "321" (ZYX) |
| pe | array | `[[x, y, z, rx, ry, rz]]`，单位：米(m)、弧度(rad) |
| pq | array | 四元数模式，不使用时留空 `[]` |

**关键**: `pe` 中的值是**相对于 follower_cart 启动时机器人位置的累积偏移**，不是增量。


### 2.3 WebSocket 状态查询 (发送 "get"，接收状态)

发送：
```
header (40B) + b"get"
```

接收 (JSON)：
```json
{
    "ret_code": 0,
    "ret_context": {
        "motion_msg": {
            "pe": [[x, y, z, rx, ry, rz]],
            "pq": [[x, y, z, qx, qy, qz, qw]],
            "actual_pos": [[j1, j2, j3, j4, j5, j6]]
        },
        "robot_msg": {
            "status": "Normal",
            "activate": "Enabled",
            "motion": "Running"
        }
    }
}
```

| 字段 | 单位 | 说明 |
|------|------|------|
| pe | mm, deg | 末端TCP位姿 (欧拉角) |
| actual_pos | deg | 6轴关节角度 |


## 3. 完整启动流程

```
步骤 1: WebSocket 连接
    ws_cmd  = connect(192.168.0.10:5888)   # 控制端口

步骤 2: 登录
    send("logout")                          # 退出已有用户
    send("login --user=Engineer --pwd=<md5>")

步骤 3: 使能
    send("manual_en")

步骤 4: 设置速度
    send("set_jog_vel --vel_percent=100")

步骤 5: 开启跟随功能
    send("start_follower")

步骤 6: 切换坐标系
    send("set_jog_coordinate --tool")       # 使用工具坐标系

步骤 7: 启动笛卡尔跟随 (不等回包)
    send("follower_cart")                   # 此刻记录"零点"

步骤 8: 连接 UDP
    udp_socket.connect(192.168.0.10:9998)

步骤 9: 循环发送位姿
    每帧: udp_send(header + json(pe))       # pe = 累积偏移

步骤 10: 停止
    send("stop_follower")
```

> 步骤 1-7 通过 WebSocket 5888 发送。
> 步骤 9 通过 UDP 9998 发送。


## 4. 位姿数据发送逻辑

### 4.1 核心概念：累积偏移

```
follower_cart 启动时，机器人记录当前位置为"零点 P0"

之后每帧发送的 pe = 目标位置相对于 P0 的总偏移

  pe = [0, 0, 0, 0, 0, 0]    → 机器人保持在零点
  pe = [0.05, 0, 0, 0, 0, 0] → 机器人移动到零点 +50mm (工具X方向)
  pe = [0.10, 0, 0, 0, 0, 0] → 机器人移动到零点 +100mm

重复发送相同的 pe → 机器人保持在那个位置不动
逐帧增大 pe → 机器人持续运动
```


### 4.2 坐标映射 (CSV 基座坐标 → 工具坐标系)

由于 `set_jog_coordinate --tool` 使 pe 在工具坐标系下解释，
而 CSV 轨迹是基座坐标系，需要做坐标变换：

```
CSV 基座坐标增量:  delta = [dx, dy, dz]  (mm)
                   转换为米: dx/1000, dy/1000, dz/1000

发送给工控机 pe:   [dy, dx, -dz, ...]
                      ↑   ↑    ↑
                      │   │    └── Z 取反
                      │   └─────── CSV_X → 工具Y
                      └─────────── CSV_Y → 工具X
```

对应代码 (`main_window.py` `_run_points_sequence_udp`):
```python
delta = np.array(p) - p_start                     # CSV 偏移 (mm, deg)
dx, dy, dz = delta[:3] / 1000.0                   # mm → m
drx, dry, drz = np.deg2rad(delta[3:])              # deg → rad

# 坐标映射后发送
udp_client.send_pose_euler(dy, dx, -dz, dry, drx, drz)
```


### 4.3 手动遥操作 (步进按钮)

手动遥操作通过 `send_raw_increment(dx_mm, dy_mm, dz_mm)` 发送，
内部维护 `follower_offset` 累积变量：

```python
# robot_controller.py
follower_offset[0] += dx_mm / 1000.0      # X 累积
follower_offset[1] += dy_mm / 1000.0      # Y 累积
follower_offset[2] += -dz_mm / 1000.0     # Z 取反后累积

udp_client.send_pose_euler(*follower_offset)  # 发送累积总量
```

> 手动遥操作的坐标系已经是工具坐标系，不需要 X/Y 互换。


## 5. 状态反馈获取

### 唯一方式：WebSocket 轮询

UDP 9998 不回包，实时位姿通过 WebSocket 获取：

```
robot_controller._monitor_loop() (独立线程, 20Hz)
    │
    ├── ws_client.send_get_status()     # 发送 "get" 到 5888
    │
    ├── 解析 motion_msg.pe             # 获取 TCP 位姿 [x,y,z,rx,ry,rz] (mm,deg)
    ├── 解析 motion_msg.actual_pos     # 获取关节角度 [j1..j6] (deg)
    │
    ├── state.update_tcp(pe)            # 缓存到 RobotState
    └── state.update_joints(joints)     # 缓存到 RobotState
```

任何地方读取位姿：
```python
tcp = controller.state.get_tcp()       # [x,y,z,rx,ry,rz] (mm, deg)
joints = controller.state.get_joints() # [j1..j6] (deg)
```


## 6. UDP 轨迹执行流水线

完整自动化流程 (`_run_udp_pipeline`)：

```
用户点击 "UDP 轨迹执行" 按钮
    │
    ├── 步骤 1: IK 对位
    │   读取 CSV 第一个点 → FastIK 求解 → move_joint 移到起点
    │
    ├── 步骤 2: 启动跟随模式
    │   start_follower → set_jog_coordinate --tool → follower_cart
    │   此刻机器人位置 = CSV第一个点 = follower零点
    │
    ├── 步骤 3: 连接 UDP
    │   udp_socket.connect(192.168.0.10:9998)
    │
    └── 步骤 4: 逐帧播放
        for each point in CSV:
            delta = point - CSV[0]           # 相对于起始点的偏移
            pe = 坐标映射(delta)              # [dy, dx, -dz, ...]
            udp_send(pe)                      # 发送累积偏移
            sleep(target_interval)            # 控制发送频率

        stop_follower                         # 播放完成，停止跟随
```


## 7. 关键文件索引

| 文件 | 职责 |
|------|------|
| `tools/follower_client.py` : `FollowerClient` | UDP 通信底层 (打包/发送/接收) |
| `tools/follower_client.py` : `FollowerClientWebSocket` | WebSocket 通信底层 (指令/状态查询) |
| `KAANH_Digital_Twin/robot_controller.py` | 通信分发门面 (连接管理/监控循环/遥操作) |
| `KAANH_Digital_Twin/main_window.py` | 轨迹调度 (`_run_udp_pipeline` / `_run_points_sequence_udp`) |
| `KAANH_Digital_Twin/core/robot_state.py` | 机器人状态缓存 (TCP/关节角/模式) |
