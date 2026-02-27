"""
机器人物理极限速度测试工具 (修正版)
用途：测量机器人在 100% 速度设定下能达到的最大实际线速度 (m/s)。
"""

import time
import math
import numpy as np
from teach_pendant.robot_controller import RobotController
from teach_pendant.signals import WorkerSignals
from PyQt5.QtCore import QCoreApplication
import sys

def test_speed():
    app = QCoreApplication(sys.argv)
    signals = WorkerSignals()
    controller = RobotController(signals)
    
    ip = "192.168.0.10"
    port = 5999
    print(f"正在连接机器人 {ip}:{port}...")
    if not controller.connect(ip, port):
        print("连接失败！")
        return

    # 必须启动监控线程，否则拿不到实时坐标反馈
    controller.start_monitoring()
    
    controller.logout()
    time.sleep(0.5)
    controller.login()
    controller.enable()
    controller.set_velocity(100)
    time.sleep(1)

    print("\n[1/3] 正在移动到测试起点...")
    home_joints = [0, -15, 105, 0, -90, 0]
    controller.move_joint(home_joints, vels=[50]*6, wait_for_finish=True)
    time.sleep(1)

    print("[2/3] 开始冲刺测试 (J1: 0 -> 30)...")
    target_joints = [30, -15, 105, 0, -90, 0] 
    
    samples = []
    start_time = time.perf_counter()
    controller.move_joint(target_joints, vels=[100]*6, wait_for_finish=False)
    
    timeout = 4.0
    while (time.perf_counter() - start_time) < timeout:
        tcp = controller.state.get_tcp() # mm
        samples.append((time.perf_counter(), np.array(tcp[:3]) / 1000.0))
        time.sleep(0.01) # 100Hz 采样
        
        curr_joints = controller.state.get_joints()
        if abs(curr_joints[0] - 30) < 0.2:
            print("检测到已到达目标位置。")
            break

    print("[3/3] 测试完成，正在分析数据...")
    if len(samples) < 10:
        print("采集数据严重不足，请检查机器人是否运行。")
        return

    velocities = []
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i-1][0]
        dist = np.linalg.norm(samples[i][1] - samples[i-1][1])
        if dt > 0.001: # 过滤采样间隔过小的情况
            velocities.append(dist / dt)

    if not velocities:
        print("无法计算速度，坐标反馈可能没有更新。")
        return

    # 过滤掉由于采集噪声产生的极高异常值
    velocities = [v for v in velocities if v < 2.0]
    
    max_speed = max(velocities) if velocities else 0
    avg_speed = sum(velocities) / len(velocities) if velocities else 0
    
    print("\n" + "="*40)
    print(f"机器人物理极限速度测试报告")
    print("-" * 40)
    print(f"有效采样点: {len(velocities)}")
    print(f"最大瞬时速度: {max_speed:.4f} m/s ({max_speed*1000:.1f} mm/s)")
    print(f"平均移动速度: {avg_speed:.4f} m/s ({avg_speed*1000:.1f} mm/s)")
    print("=" * 40)

    if max_speed < 0.01:
        print("\n[错误] 速度为0，坐标反馈未更新！请确认模拟器中机器人是否有动作。")
    elif max_speed < 0.12:
        print("\n结论建议：")
        print("当前物理极限确实太低 (约 100mm/s)。")
        print("追不上 0.1m/s 属于物理瓶颈，请在工控机调大系统速度上限。")
    else:
        print("\n结论建议：")
        print(f"物理速度充足 ({max_speed*1000:.1f} mm/s)。")
        print("追不上是因为追踪算法的'预瞄'或'提前量'参数未随速度动态增加。")

    controller.stop()

if __name__ == "__main__":
    test_speed()
