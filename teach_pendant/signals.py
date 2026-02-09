"""
Qt 信号类 - 用于线程间通信
"""

from PyQt5.QtCore import pyqtSignal, QObject


class WorkerSignals(QObject):
    """工作线程信号"""
    joints_updated = pyqtSignal(list)  # 关节位置更新
    status_updated = pyqtSignal(str)   # 状态消息
    connection_changed = pyqtSignal(bool, str)  # 连接状态变化 (connected, port_type)
    command_finished = pyqtSignal(bool, str)  # 命令执行完成 (success, message)
    error_occurred = pyqtSignal(str)   # 错误发生
    robot_status_updated = pyqtSignal(dict)  # 机器人状态更新
