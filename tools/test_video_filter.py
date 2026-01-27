import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 确保可以导入模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vision import VisionSystem
from estimator import KalmanFilterEstimator

def test_vision_filtering():
    print("启动真实视频轨迹滤波测试...")
    
    # ==========================
    # 1. 参数设置
    # ==========================
    # 覆盖 config 中的设置，方便在此处调试
    video_source = config.VISION_SOURCE
    
    # 仿真步长 (模拟主循环的时间步进)
    # 使用较小的时间步长可以模拟真实运行时的平滑效果
    sim_dt = config.DT_CTRL  # e.g. 0.002s
    
    # 视觉采样周期
    vision_dt = config.DT_VISION # e.g. 0.01s
    
    # KF 参数调试
    kf_process_noise = 0.05   # Q: 0.01 ~ 1.0 (越小越滞后但平滑，越大越灵敏但抖动)
    kf_measure_noise = 0.5 # R: 0.001 ~ 0.01 (通常设为相机测量精度，如 5mm)
    
    print(f"视频源: {video_source}")
    print(f"KF参数: Q={kf_process_noise}, R={kf_measure_noise}")
    
    # ==========================
    # 2. 初始化
    # ==========================
    try:
        vs = VisionSystem(source=video_source)
    except Exception as e:
        print(f"初始化视觉系统失败: {e}")
        return

    # 初始化滤波器
    # 注意：这里的 dt 应该是视觉更新的周期 DT_VISION，而不是仿真步长
    # 因为 predict/update 是在 vision_dt 间隔进行的
    kf = KalmanFilterEstimator(vision_dt, process_noise=kf_process_noise, measurement_noise=kf_measure_noise)
    
    duration = vs.get_duration()
    if duration <= 0:
        duration = 10.0 # 默认时长
        
    print(f"视频时长: {duration:.2f}s")
    
    # ==========================
    # 3. 运行处理循环
    # ==========================
    t_sim = 0.0
    vision_timer = 0.0
    
    # 数据记录
    time_log = []
    raw_pos_log = []   # 原始识别坐标 (带噪声)
    filt_pos_log = []  # 滤波后坐标
    filt_vel_log = []  # 滤波后速度
    filt_acc_log = []  # 滤波后加速度
    
    # 用于记录上一次有效的原始坐标，方便绘图连线（处理丢帧情况）
    last_valid_raw = None
    
    total_steps = int(duration / sim_dt)
    
    print("开始处理...")
    start_time = time.time()
    
    for step in range(total_steps):
        t_sim += sim_dt
        vision_timer += sim_dt
        
        # 模拟视觉更新周期
        if vision_timer >= vision_dt:
            vision_timer = 0.0
            
            # 获取当前时刻的视频帧数据
            pos, valid = vs.update_to_time(t_sim)
            
            # 记录数据
            current_raw = None
            if valid:
                current_raw = pos
                last_valid_raw = pos
            else:
                # 如果这一帧没识别到，可以选择记录 None 或者上一次的值
                # 为了绘图方便，这里记录 None，绘图时处理
                current_raw = np.array([np.nan, np.nan, np.nan])
            
            # KF 更新 (注意：在实际仿真中，即使没识别到，可能也需要 predict，或者 skip)
            # 这里简单起见，只有 valid 时 update，否则只 predict 或者保持
            # 为了测试滤波效果，我们假设 continuous prediction
            
            if valid:
                # 有测量值：Update
                vel, acc = kf.update(pos)
                state = kf.get_state()
                current_filt = state['pos']
            else:
                # 无测量值：Predict (纯外推)
                # 注意：estimator.update 内部逻辑是基于测量的。
                # 如果没有测量，我们通常只做 predict。
                # 但这里的 KalmanFilterEstimator 类目前只提供了 update 方法接受测量。
                # 我们可以调用 predict_future(0) 来获取当前状态，或者扩展 estimator 支持无测量 predict
                # 这里的 estimator 代码只有 update 和 predict_future(外推)。
                # 简单处理：如果没有数据，我们就不调用 update，状态保持不变(或者应该做纯预测)
                # 为了看清滤波器的行为，这里暂不更新 KF 状态，直接取当前状态
                # (真实场景中可能会用 predict 更新 P 矩阵)
                state = kf.get_state()
                current_filt = state['pos']
                vel = state['vel']
                acc = state['acc']

            # 记录日志 (按视觉帧记录，或者按仿真步记录均可，这里按视觉帧记录)
            time_log.append(t_sim)
            raw_pos_log.append(current_raw)
            filt_pos_log.append(current_filt)
            filt_vel_log.append(vel)
            filt_acc_log.append(acc)
            
        # 显示进度
        if step % 500 == 0:
            print(f"\r进度: {t_sim:.2f}/{duration:.2f}s", end="")
            
    print(f"\n处理完成，耗时: {time.time() - start_time:.2f}s")
    vs.stop()

    # ==========================
    # 4. 数据转换与绘图
    # ==========================
    time_log = np.array(time_log)
    raw_pos_log = np.array(raw_pos_log)
    filt_pos_log = np.array(filt_pos_log)
    filt_vel_log = np.array(filt_vel_log)
    filt_acc_log = np.array(filt_acc_log)
    
    # 绘图
    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(f"Vision & Filter Test\nVideo: {os.path.basename(str(video_source))}, Q={kf_process_noise}, R={kf_measure_noise}", fontsize=16)
    
    # 1. XY 平面轨迹
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(raw_pos_log[:, 0], raw_pos_log[:, 1], 'r.', markersize=2, alpha=0.5, label='Raw Vision (Measurement)')
    ax1.plot(filt_pos_log[:, 0], filt_pos_log[:, 1], 'b-', linewidth=1.5, label='Filtered (KF)')
    ax1.set_title("XY Trajectory")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()
    
    # 2. X, Y 随时间变化
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time_log, raw_pos_log[:, 0], 'r.', markersize=2, alpha=0.3, label='Raw X')
    ax2.plot(time_log, filt_pos_log[:, 0], 'b-', linewidth=1, label='Filt X')
    # ax2.plot(time_log, raw_pos_log[:, 1], 'm.', markersize=2, alpha=0.3, label='Raw Y')
    # ax2.plot(time_log, filt_pos_log[:, 1], 'c-', linewidth=1, label='Filt Y')
    ax2.set_title("X Position vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.grid(True)
    ax2.legend()
    
    # 3. 速度估计
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_log, filt_vel_log[:, 0], 'b-', label='Vx')
    ax3.plot(time_log, filt_vel_log[:, 1], 'g-', label='Vy')
    ax3.set_title("Estimated Velocity")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.grid(True)
    ax3.legend()
    
    # 4. 加速度估计 (用于前馈，最关键)
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time_log, filt_acc_log[:, 0], 'b-', label='Ax')
    ax4.plot(time_log, filt_acc_log[:, 1], 'g-', label='Ay')
    ax4.set_title("Estimated Acceleration (Feedforward)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Acceleration (m/s^2)")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vision_filtering()