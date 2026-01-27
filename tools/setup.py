import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as sm
from trajectory import TrajectoryGenerator
import config


def compute_ik(robot, target_pos, q_guess=None):
    """
    根据目标末端位置计算逆运动学
    """
    if q_guess is None:
        q_guess = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
    
    robot.q = q_guess
    T_init = robot.fkine(robot.q)
    R_fixed = T_init.R
    
    T_target = sm.SE3.Rt(R_fixed, target_pos)
    sol = robot.ikine_LM(T_target, q0=q_guess)
    
    if sol.success:
        return sol.q
    else:
        print(f"[警告] 逆运动学求解失败，使用初始猜测值")
        return q_guess


def preview_trajectory(robot_init_ee_pos=None):
    """
    预览机器人初始位置和轨迹，不运行仿真
    
    :param robot_init_ee_pos: 机器人初始末端位置 [x, y, z]
    :return: 计算得到的初始关节角度
    """
    # 1. 初始化机器人
    robot = rtb.models.UR5()
    
    if robot_init_ee_pos is None:
        robot_init_ee_pos = config.ROBOT_INIT_EE_POS
    ee_init = np.array(robot_init_ee_pos)
    
    q_init = compute_ik(robot, ee_init)
    robot.q = q_init
    ee_actual = robot.fkine(robot.q).t

    # 2. 根据模式创建轨迹生成器
    if config.TRAJ_MODE == 'csv':
        trajectory = TrajectoryGenerator(
            mode=config.TRAJ_MODE,
            init_pos=config.TRAJ_INIT_POS,
            center=config.TRAJ_CENTER,
            csv_path=config.TRAJ_CSV_PATH,
            z_height=config.TRAJ_Z_HEIGHT,
            world_offset=config.TRAJ_WORLD_OFFSET
        )
    else:
        trajectory = TrajectoryGenerator(
            mode=config.TRAJ_MODE,
            init_pos=config.TRAJ_INIT_POS,
            center=config.TRAJ_CENTER,
            radius=config.TRAJ_RADIUS,
            omega=config.TRAJ_OMEGA,
            velocity=config.TRAJ_VELOCITY
        )

    # 生成轨迹点
    num_points = int(config.SIM_DURATION / config.DT_CTRL)
    trajectory_points = []
    
    for _ in range(num_points):
        pos = trajectory.get_state()
        target_pos = pos + np.array(config.TRACKING_OFFSET)
        trajectory_points.append(target_pos)
        trajectory.update(config.DT_CTRL)
    
    trajectory_points = np.array(trajectory_points)

    # 3. 创建 3D 可视化
    env = rtb.backends.PyPlot.PyPlot()
    env.launch()
    env.add(robot)

    ax = env.ax
    ax.set_xlim([-0.2, 0.8])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0.0, 0.8])
    ax.view_init(elev=30, azim=45)

    # 4. 绘制轨迹曲线
    ax.plot(
        trajectory_points[:, 0],
        trajectory_points[:, 1],
        trajectory_points[:, 2],
        'r--', linewidth=1.5, label='Target Trajectory'
    )
    
    ax.scatter(
        [trajectory_points[0, 0]],
        [trajectory_points[0, 1]],
        [trajectory_points[0, 2]],
        c='blue', s=60, marker='o', label='Trajectory Start'
    )
    ax.scatter(
        [trajectory_points[-1, 0]],
        [trajectory_points[-1, 1]],
        [trajectory_points[-1, 2]],
        c='orange', s=60, marker='s', label='Trajectory End'
    )

    ax.scatter(
        [ee_actual[0]], [ee_actual[1]], [ee_actual[2]],
        c='green', s=80, marker='^', label='Robot Init EE'
    )

    # 5. 打印信息
    print(f"[预览] 轨迹模式: {config.TRAJ_MODE}")
    print(f"[预览] 轨迹点数: {num_points}")
    print(f"[预览] 目标初始末端位置: {ee_init}")
    print(f"[预览] 实际初始末端位置: {ee_actual}")
    print(f"[预览] 初始关节角度: {q_init}")
    print(f"[预览] 轨迹起点: {trajectory_points[0]}")
    print(f"[预览] 轨迹终点: {trajectory_points[-1]}")
    
    # 打印轨迹详细信息
    traj_info = trajectory.get_trajectory_info()
    print(f"[预览] 轨迹详情: {traj_info}")

    plt.legend()
    plt.title(f"Trajectory Preview (Mode: {config.TRAJ_MODE})")
    plt.show(block=False)
    env.step(0.0)
    plt.pause(0.5)
    env.hold()
    
    return q_init


def debug_preview(frames=(0, -1)):
    """调试用：打印指定帧的目标点与末端位置"""
    from simulation import run_simulation
    results = run_simulation()
    target_pos = np.array(results['target_pos'])
    q_traj = np.array(results['q'])
    robot = rtb.models.UR5()

    total = len(q_traj)
    print(f"[预览] KP={config.KP}, KI={config.KI}, TRAJ_MODE={config.TRAJ_MODE}")
    for f in frames:
        idx = f if f >= 0 else total + f
        idx = max(0, min(idx, total - 1))
        robot.q = q_traj[idx]
        ee_pos = robot.fkine(robot.q).t
        print(f"  帧 {idx:4d}: target = {target_pos[idx]}, ee = {ee_pos}")


if __name__ == "__main__":
    preview_trajectory()