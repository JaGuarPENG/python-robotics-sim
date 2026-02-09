# 遥操作 Follower Demo 技术文档

## 1. 概述

遥操作系统通过 UDP 协议向机器人控制器发送位姿指令，实现远程控制机器人运动。

## 2. 通信架构

```
┌─────────────┐      UDP 9998 (笛卡尔)      ┌─────────────┐
│   上位机     │ ─────────────────────────→ │   机器人     │
│  (Python)   │      UDP 9999 (关节)        │   控制器    │
│             │ ←───────────────────────── │             │
└─────────────┘      状态反馈 (JSON)        └─────────────┘
```

**端口分配：**
| 端口 | 用途 | 数据格式 |
|------|------|----------|
| 9998 | 笛卡尔空间跟随 | JSON |
| 9999 | 关节空间跟随 | 字符串命令 |

## 3. 发送数据格式

### 3.1 笛卡尔空间指令 (端口 9998)

**JSON 格式：**
```json
{
    "type": "321",
    "pe": [[x, y, z, rx, ry, rz]],
    "pq": []
}
```

**字段说明：**
| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 欧拉角类型：`"321"` / `"123"` / `"313"` |
| `pe` | 二维数组 | 位姿 [x, y, z, rx, ry, rz]，**单位：米和弧度** |
| `pq` | 二维数组 | 四元数位姿 [x, y, z, qx, qy, qz, qw]（与 pe 二选一） |

**pe 数组含义：**
```
pe = [x, y, z, rx, ry, rz]
      │  │  │   │   │   │
      │  │  │   │   │   └── 绕 Z 轴旋转 (弧度)
      │  │  │   │   └────── 绕 Y 轴旋转 (弧度)
      │  │  │   └────────── 绕 X 轴旋转 (弧度)
      │  │  └────────────── Z 位置 (米)
      │  └───────────────── Y 位置 (米)
      └──────────────────── X 位置 (米)
```

**重要：pe 中的值是增量值，不是绝对位置！**

### 3.2 关节空间指令 (端口 9999)

**字符串格式：**
```
jp --motor_pos=[j1, j2, j3, j4, j5, j6]
```

**示例：**
```
jp --motor_pos=[0,-5,90,10,-100,0]
```

**注意：关节指令单位是度和毫米，值是绝对位置！**

## 4. 接收数据格式 (状态反馈)

发送 `get` 指令可获取机器人状态，返回 JSON 格式：

```json
{
    "ret_code": 0,
    "ret_msg": "",
    "ret_context": {
        "follower_msg": {
            "running_state": false,
            "follower_mode": "None"
        },
        "motion_msg": {
            "actual_pos": [[j1, j2, j3, j4, j5, j6]],
            "pe": [[x, y, z, rx, ry, rz]],
            "pq": [[x, y, z, qx, qy, qz, qw]]
        }
    }
}
```

**关键字段：**
| 字段 | 说明 |
|------|------|
| `follower_msg.running_state` | 是否正在跟随 |
| `follower_msg.follower_mode` | 跟随模式：笛卡尔/关节/None |
| `motion_msg.actual_pos` | 当前关节角度 |
| `motion_msg.pe` | 当前末端位姿 (欧拉角) |
| `motion_msg.pq` | 当前末端位姿 (四元数) |

## 5. 控制指令

### 5.1 跟随控制

| 指令 | 说明 |
|------|------|
| `start_follower` | 开始跟随功能 |
| `stop_follower` | 关闭跟随功能 |
| `follower_aj` | 运行关节跟随 |
| `follower_cart` | 运行笛卡尔跟随 |

### 5.2 登录登出

```bash
# 登录
login --user=Engineer --pwd=000000

# 登出
logout
```

### 5.3 使能控制

```bash
# 虚拟模式
manual_en    # 使能
manual_ds    # 掉使能

# 真机模式
en           # 使能
ds           # 掉使能
```

### 5.4 调速

```bash
# 设置自动模式速度百分比
set_pgm_vel --vel_percent=60

# 设置手动模式速度百分比
set_jog_vel --vel_percent=60
```

## 6. C++ Demo 代码解析

### 6.1 文件结构

```
follower_demo/
├── main.cpp      # 主程序
├── plan.hpp      # Client 类声明
├── plan.cpp      # Client 类实现
├── json.hpp      # JSON 库
└── CMakeLists.txt
```

### 6.2 核心代码

```cpp
// 创建 UDP 客户端，连接到 192.168.192.149:9998
Client client("client", "192.168.192.149", "9998", aris::core::Socket::Type::UDP);

// 构造笛卡尔位姿指令
nlohmann::json j;
std::vector<std::vector<double>> pes, pqs;
std::vector<double> pe{ 0.10, 0, 0, 0, 0, 0 };  // X 方向移动 0.1 米
pes.push_back(pe);

j["type"] = "321";   // 欧拉角类型
j["pe"] = pes;       // 位姿数据
j["pq"] = pqs;       // 四元数（空）

// 发送指令
msg.copy(j.dump());
client.socket().sendMsg(msg);
```

### 6.3 接收回调

```cpp
auto onReceivedMsg = [](aris::core::Socket* socket, aris::core::Msg& msg)->int {
    std::cout << msg.data() << std::endl;  // 打印接收到的状态数据
    return 0;
};
client.socket().setOnReceivedMsg(onReceivedMsg);
```

## 7. Python 实现对比

### 7.1 当前 follower_client.py 的问题

```python
# 当前实现
def send_pose_euler(self, x, y, z, rx, ry, rz):
    j = {
        "type": "321",
        "pe": [[x, y, z, rx, ry, rz]],  # 发送的是绝对位置
        "pq": []
    }
```

**问题：**
1. 官方文档说 pe 是**增量值**，但当前代码发送的是绝对位置
2. 没有解析接收到的机器人实际位置用于误差计算

### 7.2 正确的闭环控制流程

```
1. 发送 get 指令获取当前位置
2. 计算增量 = 目标位置 - 当前位置
3. 发送增量指令
4. 接收反馈，获取实际位置
5. 计算误差 = 目标位置 - 实际位置
```

## 8. XML 配置说明

```xml
<!-- 笛卡尔跟随配置，端口 9998 -->
<FollowerCartParam tg_id="1">
    <SocketServer connect_type="UDP" port="9998" />
</FollowerCartParam>

<!-- 关节跟随配置，端口 9999 -->
<FollowerAjParam tg_id="0">
    <SocketServer connect_type="UDP" port="9999" />
</FollowerAjParam>
```

## 9. 速度限制

| 模式 | 最大速度 | 调速方式 |
|------|----------|----------|
| 笛卡尔跟随 | 点动笛卡尔设置的速度 | 最大速度 × 速度百分比 |
| 关节跟随 | 关节最大速度的 10% | 最大速度 × 速度百分比 |

## 10. 总结

### 发送什么？
- **笛卡尔模式**：JSON 格式的位姿增量 `{"type":"321", "pe":[[dx,dy,dz,drx,dry,drz]], "pq":[]}`
- **关节模式**：字符串命令 `jp --motor_pos=[j1,j2,j3,j4,j5,j6]`

### 接收什么？
- JSON 格式的机器人状态，包含：
  - `motion_msg.pe`: 当前末端位姿
  - `motion_msg.actual_pos`: 当前关节角度
  - `follower_msg.running_state`: 跟随状态

### 如何做真正的误差分析？
1. 定期发送 `get` 指令获取 `motion_msg.pe`（实际位置）
2. 误差 = 目标位置 - `motion_msg.pe`
