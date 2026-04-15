"""
示教器配置参数
"""

# 机器人连接配置
ROBOT_IP = "192.168.1.10"
PORT_CONTROL = 5999  # 控制端口 (WebSocket, 负责发送指令)
PORT_MONITOR = 5888  # 监控端口 (WebSocket, 负责读取状态)

# 预设位置
PRESET_POSITIONS = {
    "零位": [0, 0, 0, 0, 0, 0],
    "快速定位1": [0, -15, 105, 0, -90, 0],
    "位置2": [30, -45, 60, 0, -30, 45],
    "位置3(测试)": [0, -45, 90, 0, -45, 0],  # 简单姿态用于测试，末端接近垂直向下
}

# 默认运动速度参数
DEFAULT_VEL = [100, 200, 100]

# Jog 步进角度 (度)
JOG_STEP_SMALL = 1.0
JOG_STEP_MEDIUM = 5.0
JOG_STEP_LARGE = 10.0
