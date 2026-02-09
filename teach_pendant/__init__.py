"""
机器人示教器 GUI 模块

功能：
1. 连接机器人控制器 (WebSocket 5999端口)
2. 登录和使能
3. 实时显示6个关节位置
4. 3D机械臂可视化
5. 控制各关节运动 (Jog模式)
6. 预设位置快捷移动
7. 速度调节
8. 跟随模式控制 (start_follower, set_jog_coordinate, follower_cart)
9. 坐标系切换 (Joint/Tool)
10. 机器人状态查询 (运行状态、激活状态、运动状态、错误信息)

使用方法:
    python -m teach_pendant
    或
    python teach_pendant.py
"""

from .config import (
    ROBOT_IP, PORT_CONTROL, PORT_MONITOR, PRESET_POSITIONS,
    DEFAULT_VEL, JOG_STEP_SMALL, JOG_STEP_MEDIUM, JOG_STEP_LARGE
)
from .signals import WorkerSignals
from .robot_3d_widget import Robot3DWidget
from .robot_controller import RobotController
from .main_window import TeachPendantWindow
from .app import main

__all__ = [
    'ROBOT_IP', 'PORT_CONTROL', 'PORT_MONITOR', 'PRESET_POSITIONS',
    'DEFAULT_VEL', 'JOG_STEP_SMALL', 'JOG_STEP_MEDIUM', 'JOG_STEP_LARGE',
    'WorkerSignals',
    'Robot3DWidget',
    'RobotController',
    'TeachPendantWindow',
    'main',
]
