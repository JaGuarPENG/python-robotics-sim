# simulation.py
import collections
import numpy as np
import time
import roboticstoolbox as rtb
import spatialmath as sm
from config import *
from trajectory import TrajectoryGenerator
from estimator import KalmanFilterEstimator
from controller import SimplePBVS
from tools.setup import compute_ik
from vision import VisionSystem

def get_simulation_duration(trajectory=None, vision_system=None, use_real_vision=False):
    if use_real_vision and vision_system is not None:
        duration = vision_system.get_duration()
        if duration > 0:
            print(f"[Simulation] 使用视频时长: {duration:.2f}s")
            return duration
    elif not use_real_vision and trajectory is not None:
        duration = trajectory.get_duration()
        if duration is not None:
            return duration
    
    print(f"[Simulation] 使用配置固定时长: {SIM_DURATION}s")
    return SIM_DURATION

def run_simulation(show_progress=True, robot_init_q=None, robot_init_ee_pos=None, use_real_vision=False):
    # --- 1. 机器人初始化 ---
    robot = rtb.models.UR5()
    if robot_init_q is not None:
        robot.q = np.array(robot_init_q)
    elif robot_init_ee_pos is not None:
        robot.q = compute_ik(robot, np.array(robot_init_ee_pos))
    else:
        robot.q = compute_ik(robot, np.array(ROBOT_INIT_EE_POS))
    
    T_init = robot.fkine(robot.q)
    R_fixed = T_init.R 

    # --- 2. 资源初始化 ---
    trajectory = None
    vision_system = None
    
    if use_real_vision:
        # 初始化视觉系统 (此时不启动线程，改为被动调用)
        vision_system = VisionSystem(source=VISION_SOURCE)
        print(f"[Simulation] 真实视觉模式 (源: {VISION_SOURCE})")
    else:
        # 初始化虚拟轨迹
        if TRAJ_MODE == 'csv':
            trajectory = TrajectoryGenerator(mode=TRAJ_MODE, csv_path=TRAJ_CSV_PATH, 
                                             z_height=TRAJ_Z_HEIGHT, world_offset=TRAJ_WORLD_OFFSET, init_pos=TRAJ_INIT_POS, center=TRAJ_CENTER)
        else:
            trajectory = TrajectoryGenerator(mode=TRAJ_MODE, init_pos=TRAJ_INIT_POS, center=TRAJ_CENTER,
                                             radius=TRAJ_RADIUS, omega=TRAJ_OMEGA, velocity=TRAJ_VELOCITY)
    
    sim_duration = get_simulation_duration(trajectory, vision_system, use_real_vision)
    
    # 状态估计器 & 控制器
    estimator = KalmanFilterEstimator(DT_VISION, process_noise=KF_PROCESS_NOISE, measurement_noise=KF_MEASUREMENT_NOISE, vel_cov=KF_VEL_COV, acc_cov=KF_ACC_COV)
    controller = SimplePBVS(KP, KI, MAX_STEP, INTEGRAL_LIMIT)

    lag_frames = max(1, int(round(SENSE_DELAY / DT_VISION)))
    obs_buffer = collections.deque(maxlen=lag_frames + 1)

    # 变量初始化
    t_sim = 0.0
    vision_timer = 0.0
    time_since_last_vision = 0.0
    latest_delayed_pos = None
    real_obj_pos = None # 用于存储当前帧的真值(或测量值)

    # 数据记录
    results = {'time': [], 'error': [], 'robot_pos': [], 'target_pos': [], 'q': []}

    total_steps = int(sim_duration / DT_CTRL)
    print(f"开始仿真: 时长 {sim_duration:.2f}s, 总步数 {total_steps}")
    
    try:
        for step in range(total_steps):
            # --- A. 时间推进 ---
            t_sim += DT_CTRL
            vision_timer += DT_CTRL
            
            # --- B. 物理环境更新 (获取目标位置) ---
            if not use_real_vision:
                real_obj_pos = trajectory.get_state()
                trajectory.update(DT_CTRL)
            
            # --- C. 视觉系统更新 (模拟采样或读取视频) ---
            if vision_timer >= DT_VISION:
                vision_timer = 0.0 # 重置计时器
                
                measured_pos = None
                
                if use_real_vision:
                    # 【关键】同步读取视频：请求 t_sim 时刻的数据
                    pos, valid = vision_system.update_to_time(t_sim)
                    
                    if valid:
                        measured_pos = pos
                        real_obj_pos = pos # 在真实模式下，视测量值为真值用于记录
                    else:
                        # 视频结束或没识别到
                        if pos is None and t_sim > sim_duration: break
                else:
                    measured_pos = real_obj_pos
                    measured_pos[2] = real_obj_pos[2]

                # --- 存入缓冲区 & KF更新 ---
                if measured_pos is not None:
                    # obs_buffer.append(measured_pos)
                    # # 模拟传输延迟
                    # delayed_pos = obs_buffer[0] if len(obs_buffer) >= lag_frames + 1 else obs_buffer[0]
                    
                    estimator.update(measured_pos)
                    time_since_last_vision = 0.0 # 重置外推计时
                    
                    # 获取用于控制的平滑基准点
                    state = estimator.get_state()
                    latest_delayed_pos = state['pos']
                    
                    # Z轴处理
                    if use_real_vision:
                        latest_delayed_pos[2] = measured_pos[2]
                    elif TRAJ_MODE == 'csv':
                        latest_delayed_pos[2] = TRAJ_Z_HEIGHT
                    else:
                        latest_delayed_pos[2] = real_obj_pos[2]
            else:
                time_since_last_vision += DT_CTRL

            # --- D. 控制器计算 ---
            if latest_delayed_pos is not None:
                # 预测当前时刻位置
                total_pred_time = SENSE_DELAY + time_since_last_vision + ESTIMATED_SERVO_LAG
                pred_pos, pred_vel = estimator.predict_future(total_pred_time)
                pred_pos[2] = latest_delayed_pos[2] # 保持 Z
                
                desired_robot_pos = pred_pos + np.array(TRACKING_OFFSET)
                
                # PBVS
                curr_robot_pos = robot.fkine(robot.q).t
                next_robot_pos = controller.compute_next_pose(curr_robot_pos, desired_robot_pos, pred_vel, DT_CTRL)
                
                # IK & Servo Lag
                T_next = sm.SE3.Rt(R_fixed, next_robot_pos)
                sol = robot.ikine_LM(T_next, q0=robot.q)
                if sol.success:
                    robot.q = SERVO_ALPHA * sol.q + (1 - SERVO_ALPHA) * robot.q

            # --- E. 数据记录 ---
            # 仅当当前时刻有有效目标时记录
            if real_obj_pos is not None:
                true_desired = real_obj_pos + np.array(TRACKING_OFFSET)
                curr_pos = robot.fkine(robot.q).t
                error = np.linalg.norm(curr_pos - true_desired)
                
                results['time'].append(t_sim)
                results['error'].append(error)
                results['robot_pos'].append(curr_pos)
                results['target_pos'].append(true_desired)
                results['q'].append(robot.q.copy())

            # 打印进度
            if show_progress and step % 50 == 0:
                err_str = f"{error*1000:.2f}" if 'error' in locals() else "Wait..."
                print(f"\r进度: {step/total_steps*100:.1f}% | T: {t_sim:.2f}s | Err: {err_str}mm", end="")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if vision_system: vision_system.stop()

    print(f"\n仿真结束")
    return results