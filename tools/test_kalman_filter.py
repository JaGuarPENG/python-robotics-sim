import numpy as np
import matplotlib.pyplot as plt
import os

# 动态导入 config，避免循环导入
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimator import KalmanFilterEstimator


def get_ground_truth(mode, t, **kwargs):
    """
    生成真实轨迹数据
    返回: true_x, true_y, true_vx, true_vy, true_ax, true_ay
    """
    steps = len(t)
    
    if mode == 'circle':
        # 圆周运动配置
        radius = kwargs.get('radius', 0.15)
        omega = kwargs.get('omega', 0.8)
        center_x = kwargs.get('center_x', 0.4)
        center_y = kwargs.get('center_y', 0.0)
        
        x = center_x + radius * np.cos(omega * t)
        y = center_y + radius * np.sin(omega * t)
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        ax = -radius * omega**2 * np.cos(omega * t)
        ay = -radius * omega**2 * np.sin(omega * t)
        
    elif mode == 'linear':
        # 直线运动配置
        start_x, start_y = 0.2, -0.2
        vel_x = kwargs.get('vel_x', 0.1) # X轴匀速 0.1 m/s
        vel_y = kwargs.get('vel_y', 0.05) # Y轴匀速 0.05 m/s
        
        x = start_x + vel_x * t
        y = start_y + vel_y * t
        
        # 匀速运动，速度恒定
        vx = np.full(steps, vel_x)
        vy = np.full(steps, vel_y)
        
        # 匀速运动，加速度为 0
        ax = np.zeros(steps)
        ay = np.zeros(steps)
        
    return x, y, vx, vy, ax, ay

def test_kalman_filter():
    # ==========================
    # 0. 测试模式选择
    # ==========================
    # 可选: 'circle' (圆周) 或 'linear' (直线)
    TEST_MODE = 'linear' 
    
    print(f"启动卡尔曼滤波测试... 当前模式: {TEST_MODE}")
    
    # ==========================
    # 1. 仿真参数设置
    # ==========================
    dt = 0.033           
    total_time = 4.0     
    steps = int(total_time / dt)
    t = np.linspace(0, total_time, steps)
    
    # 噪声设置 (2mm 标准差)
    real_noise_std = 0.002 
    
    # KF 参数 (使用了你觉得效果不错的参数)
    kf_process_noise = 0.05   # Q
    kf_measure_noise = 0.5    # R

    kf = KalmanFilterEstimator(dt, process_noise=kf_process_noise, measurement_noise=kf_measure_noise)

    # ==========================
    # 2. 获取真值 (Ground Truth)
    # ==========================
    true_x, true_y, true_vx, true_vy, true_ax, true_ay = get_ground_truth(
        TEST_MODE, t, 
        radius=0.15, omega=0.8,       # 圆周参数
        vel_x=0.15, vel_y=0.08        # 直线参数
    )

    # ==========================
    # 3. 生成测量数据 (叠加噪声)
    # ==========================
    np.random.seed(42)
    meas_x = true_x + np.random.normal(0, real_noise_std, steps)
    meas_y = true_y + np.random.normal(0, real_noise_std, steps)

    # ==========================
    # 4. 运行滤波器
    # ==========================
    est_pos_list = []
    est_vel_list = []
    est_acc_list = []
    
    for i in range(steps):
        z = [meas_x[i], meas_y[i]]
        
        # KF Update
        vel_est, acc_est = kf.update(z)
        state = kf.get_state()
        
        est_pos_list.append(state['pos'])
        est_vel_list.append(vel_est)
        est_acc_list.append(acc_est)

    est_pos = np.array(est_pos_list)
    est_vel = np.array(est_vel_list)
    est_acc = np.array(est_acc_list)

    # ==========================
    # 5. 可视化分析
    # ==========================
    plt.figure(figsize=(14, 10))
    plt.suptitle(f'KF Test [{TEST_MODE.upper()}]\nQ={kf_process_noise}, R={kf_measure_noise}, Noise={real_noise_std*1000}mm', fontsize=16)
    
    # 图1: 2D 轨迹
    plt.subplot(2, 2, 1)
    plt.plot(true_x, true_y, 'g-', linewidth=2, label='Ground Truth')
    plt.plot(meas_x, meas_y, 'r.', alpha=0.3, label='Measurement')
    plt.plot(est_pos[:,0], est_pos[:,1], 'b-', linewidth=2, label='KF Estimate')
    plt.title('2D Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 图2: 位置误差
    plt.subplot(2, 2, 2)
    error_dist = np.sqrt((est_pos[:,0] - true_x)**2 + (est_pos[:,1] - true_y)**2)
    meas_error_dist = np.sqrt((meas_x - true_x)**2 + (meas_y - true_y)**2)
    
    plt.plot(t, meas_error_dist * 1000, 'r-', alpha=0.2, label='Raw Meas Error')
    plt.plot(t, error_dist * 1000, 'b-', linewidth=2, label='KF Error')
    plt.title('Position Error (mm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (mm)')
    plt.legend()
    plt.grid(True)
    
    print(f"[{TEST_MODE}] 平均测量误差: {np.mean(meas_error_dist)*1000:.3f} mm")
    print(f"[{TEST_MODE}] 平均滤波误差: {np.mean(error_dist)*1000:.3f} mm")

    # 图3: 速度估计 (X轴)
    plt.subplot(2, 2, 3)
    plt.plot(t, true_vx, 'g--', label='True Vx')
    plt.plot(t, est_vel[:,0], 'b-', label='Est Vx')
    plt.title('Velocity Estimate (X-axis)')
    plt.ylabel('m/s')
    plt.legend()
    plt.grid(True)

    # 图4: 加速度估计 (X轴)
    plt.subplot(2, 2, 4)
    plt.plot(t, true_ax, 'g--', label='True Ax')
    plt.plot(t, est_acc[:,0], 'b-', label='Est Ax')
    plt.title('Acceleration Estimate (X-axis)')
    plt.ylabel('m/s^2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_kalman_filter()