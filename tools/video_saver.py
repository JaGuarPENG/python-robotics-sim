import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import roboticstoolbox as rtb
import time
import os

# 动态导入 config，避免循环导入
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VideoSaver:
    """
    用于保存和回放仿真视频的工具类
    """
    
    def __init__(self):
        """初始化 VideoSaver"""
        self.robot = None
        self.env = None
    
    def _init_robot(self):
        """初始化机器人模型"""
        robot = rtb.models.UR5()
        robot.eeframe = False
        robot.jointaxes = False
        return robot
    
    def save_simulation_video(self, results, filename='output/simulation.mp4', fps=30, speed_factor=1.0):
        """
        保存仿真动画为视频文件 (简化版，只显示轨迹点)
        
        :param results: 仿真结果字典
        :param filename: 输出文件名 (支持 .mp4, .gif, .avi)
        :param fps: 视频帧率
        :param speed_factor: 播放速度倍率 (1.0 = 真实速度)
        """
        print(f"\n正在生成视频: {filename}")
        
        # 1. 初始化机器人
        robot = self._init_robot()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置视角
        ax.set_xlim([-0.2, 0.6])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([0.0, 0.6])
        ax.view_init(elev=20, azim=-45)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 2. 提取数据
        target_pos = np.array(results['target_pos'])
        q_traj = results['q']
        sim_time = results['time']
        total_frames = len(q_traj)
        
        # 计算视频帧数 - 按真实时间计算
        sim_duration = sim_time[-1] - sim_time[0]
        video_duration = sim_duration / speed_factor  # 视频实际时长
        num_video_frames = int(video_duration * fps)
        
        # 创建视频帧索引映射
        video_frame_times = np.linspace(sim_time[0], sim_time[-1], num_video_frames)
        video_frame_indices = np.searchsorted(sim_time, video_frame_times)
        video_frame_indices = np.clip(video_frame_indices, 0, total_frames - 1)
        
        print(f"仿真帧数: {total_frames}, 视频帧数: {num_video_frames}")
        print(f"仿真时长: {sim_duration:.2f}s, 视频时长: {video_duration:.2f}s, FPS: {fps}")
        
        # 3. 绘制静态元素
        ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
                'r--', linewidth=1.5, alpha=0.7, label='Target Path')
        
        # 初始化动态元素
        target_marker, = ax.plot([], [], [], 'ro', markersize=8, label='Target')
        robot_marker, = ax.plot([], [], [], 'go', markersize=8, label='Robot EE')
        robot_trail, = ax.plot([], [], [], 'g-', linewidth=1, alpha=0.5)
        
        # 时间文本
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        error_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        
        ax.legend(loc='upper right')
        
        # 存储机器人轨迹
        robot_trail_x, robot_trail_y, robot_trail_z = [], [], []
        
        # 4. 动画更新函数
        def update(frame):
            nonlocal robot_trail_x, robot_trail_y, robot_trail_z
            
            sim_idx = video_frame_indices[frame]
            
            # 更新机器人关节
            robot.q = q_traj[sim_idx]
            ee_pos = robot.fkine(robot.q).t
            
            # 更新目标位置
            tp = target_pos[sim_idx]
            target_marker.set_data([tp[0]], [tp[1]])
            target_marker.set_3d_properties([tp[2]])
            
            # 更新机器人末端位置
            robot_marker.set_data([ee_pos[0]], [ee_pos[1]])
            robot_marker.set_3d_properties([ee_pos[2]])
            
            # 更新机器人轨迹
            robot_trail_x.append(ee_pos[0])
            robot_trail_y.append(ee_pos[1])
            robot_trail_z.append(ee_pos[2])
            robot_trail.set_data(robot_trail_x, robot_trail_y)
            robot_trail.set_3d_properties(robot_trail_z)
            
            # 更新文本
            current_time = sim_time[sim_idx]
            error = np.linalg.norm(ee_pos - tp) * 1000  # mm
            time_text.set_text(f'Time: {current_time:.2f}s / {sim_time[-1]:.2f}s')
            error_text.set_text(f'Error: {error:.2f} mm')
            
            # 进度显示
            if frame % 50 == 0:
                print(f"\r生成进度: {frame}/{num_video_frames} ({100*frame/num_video_frames:.1f}%)", end="")
            
            return target_marker, robot_marker, robot_trail, time_text, error_text
        
        # 5. 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=num_video_frames,
            interval=1000/fps, blit=False
        )
        
        # 6. 保存视频
        self._save_animation(ani, filename, fps)
        plt.close(fig)

    def save_simulation_video_with_robot(self, results, filename='output/simulation_robot.mp4', fps=30, speed_factor=1.0):
        """
        保存带有完整机器人模型的仿真视频 (使用 roboticstoolbox 的渲染)
        
        :param results: 仿真结果字典
        :param filename: 输出文件名
        :param fps: 视频帧率
        :param speed_factor: 播放速度倍率 (1.0 = 真实速度)
        """
        print(f"\n正在生成带机器人模型的视频: {filename}")
        
        # 1. 初始化机器人和环境
        robot = self._init_robot()
        
        env = rtb.backends.PyPlot.PyPlot()
        env.launch()
        env.add(robot)
        
        fig = env.fig
        ax = env.ax
        
        # 设置视角
        ax.set_xlim([-0.2, 0.6])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([0.0, 0.6])
        ax.view_init(elev=20, azim=-45)
        
        # 2. 提取数据
        target_pos = np.array(results['target_pos'])
        q_traj = results['q']
        sim_time = results['time']
        total_frames = len(q_traj)
        
        # 计算视频帧数 - 按真实时间计算
        sim_duration = sim_time[-1] - sim_time[0]
        video_duration = sim_duration / speed_factor
        num_video_frames = int(video_duration * fps)
        
        # 创建视频帧索引映射
        video_frame_times = np.linspace(sim_time[0], sim_time[-1], num_video_frames)
        video_frame_indices = np.searchsorted(sim_time, video_frame_times)
        video_frame_indices = np.clip(video_frame_indices, 0, total_frames - 1)
        
        print(f"仿真帧数: {total_frames}, 视频帧数: {num_video_frames}")
        print(f"仿真时长: {sim_duration:.2f}s, 视频时长: {video_duration:.2f}s, FPS: {fps}")
        
        # 3. 绘制静态元素
        ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
                'r--', linewidth=1.5, alpha=0.7, label='Target Path')
        
        target_marker, = ax.plot([], [], [], 'ro', markersize=8, label='Target')
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        error_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        
        ax.legend(loc='upper right')
        
        # 4. 动画更新函数
        def update(frame):
            sim_idx = video_frame_indices[frame]
            
            # 更新机器人关节
            robot.q = q_traj[sim_idx]
            
            # 更新目标位置
            tp = target_pos[sim_idx]
            target_marker.set_data([tp[0]], [tp[1]])
            target_marker.set_3d_properties([tp[2]])
            
            # 计算误差
            ee_pos = robot.fkine(robot.q).t
            error = np.linalg.norm(ee_pos - tp) * 1000
            
            # 更新文本
            current_time = sim_time[sim_idx]
            time_text.set_text(f'Time: {current_time:.2f}s / {sim_time[-1]:.2f}s')
            error_text.set_text(f'Error: {error:.2f} mm')
            
            # 更新机器人显示
            env.step(dt=0.001)
            
            # 进度显示
            if frame % 50 == 0:
                print(f"\r生成进度: {frame}/{num_video_frames} ({100*frame/num_video_frames:.1f}%)", end="")
            
            return target_marker, time_text, error_text
        
        # 5. 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=num_video_frames,
            interval=1000/fps, blit=False
        )
        
        # 6. 保存视频
        self._save_animation(ani, filename, fps)
        plt.close(fig)

    def replay_simulation(self, results, realtime=True, speed_factor=1.0, lightweight=True):
        """
        回放仿真结果

        :param results: 仿真结果字典
        :param realtime: 是否按真实时间播放
        :param speed_factor: 播放速度倍率 (1.0 = 真实速度, 2.0 = 2倍速, 0.5 = 半速)
        :param lightweight: 是否使用轻量级模式 (不渲染机器人模型，速度更快)
        """
        if lightweight:
            return self._replay_lightweight(results, realtime, speed_factor)

        print("\n正在初始化 3D 动画环境...")
        
        # 1. 开启交互模式
        plt.ion()
        
        # 2. 初始化环境
        robot = self._init_robot()
        
        env = rtb.backends.PyPlot.PyPlot()
        env.launch() 
        env.add(robot)
        
        # 3. 设置视角
        ax = env.ax
        ax.set_xlim([-0.2, 0.6])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([0.0, 0.6])
        ax.view_init(elev=20, azim=-45)

        # 4. 绘制轨迹
        target_pos = np.array(results['target_pos'])
        robot_pos = np.array(results['robot_pos'])
        
        # 绘制目标轨迹和机器人轨迹
        ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
                'r--', linewidth=1, alpha=0.5, label='Target Path')
        
        # 动态元素
        target_marker, = ax.plot([target_pos[0, 0]], [target_pos[0, 1]], [target_pos[0, 2]], 
                                  'ro', markersize=8, label='Target')
        robot_trail, = ax.plot([], [], [], 'g-', linewidth=2, alpha=0.7, label='Robot Trail')

        # 添加文本显示
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        error_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        speed_text = ax.text2D(0.02, 0.85, f'Speed: {speed_factor}x', transform=ax.transAxes, fontsize=10)
        
        ax.legend(loc='upper right')

        # 5. 在循环开始前，强制弹窗且不阻塞
        plt.show(block=False)
        plt.pause(0.5)

        q_traj = results['q']
        sim_time = results['time']
        error_data = results['error']
        total_frames = len(q_traj)
        
        print(f"开始回放... (总帧数: {total_frames}, 仿真时长: {sim_time[-1]:.2f}s)")
        print(f"播放速度: {speed_factor}x, 实时模式: {realtime}")
        
        # 存储机器人轨迹用于绘制
        trail_x, trail_y, trail_z = [], [], []
        
        start_real_time = time.time()
        start_sim_time = sim_time[0]
        last_display_time = start_real_time
        display_interval = 1.0 / 60  # 目标 60 FPS 显示刷新率
        
        try:
            frame_idx = 0
            while frame_idx < total_frames:
                current_real_time = time.time()
                
                if realtime:
                    # 按真实时间计算当前应该显示的仿真时间
                    elapsed_real_time = current_real_time - start_real_time
                    target_sim_time = start_sim_time + elapsed_real_time * speed_factor
                    
                    # 找到对应的帧索引
                    while frame_idx < total_frames - 1 and sim_time[frame_idx] < target_sim_time:
                        frame_idx += 1
                    
                    # 如果显示太快，等待
                    if sim_time[frame_idx] > target_sim_time:
                        wait_time = (sim_time[frame_idx] - target_sim_time) / speed_factor
                        if wait_time > 0.001:
                            time.sleep(min(wait_time, 0.1))  # 最多等待 0.1 秒
                            continue
                else:
                    # 非实时模式：固定跳帧
                    frame_idx += max(1, int(total_frames / 300))  # 约 300 帧显示
                
                if frame_idx >= total_frames:
                    break
                
                # 控制显示刷新率
                if current_real_time - last_display_time < display_interval:
                    continue
                last_display_time = current_real_time
                
                # 更新目标点位置
                tp = target_pos[frame_idx]
                target_marker.set_data([tp[0]], [tp[1]])
                target_marker.set_3d_properties([tp[2]])

                # 更新机器人关节
                robot.q = q_traj[frame_idx]
                
                # 更新机器人轨迹
                ee_pos = robot.fkine(robot.q).t
                trail_x.append(ee_pos[0])
                trail_y.append(ee_pos[1])
                trail_z.append(ee_pos[2])
                robot_trail.set_data(trail_x, trail_y)
                robot_trail.set_3d_properties(trail_z)
                
                # 更新文本
                current_sim_time = sim_time[frame_idx]
                current_error = error_data[frame_idx] * 1000  # 转换为 mm
                time_text.set_text(f'Time: {current_sim_time:.2f}s / {sim_time[-1]:.2f}s')
                error_text.set_text(f'Error: {current_error:.2f} mm')
                
                # 刷新显示
                env.step(dt=0.001)
                plt.pause(0.001)
                
        except KeyboardInterrupt:
            print("\n用户中断")

        # 显示最终帧
        robot.q = q_traj[-1]
        tp = target_pos[-1]
        target_marker.set_data([tp[0]], [tp[1]])
        target_marker.set_3d_properties([tp[2]])
        time_text.set_text(f'Time: {sim_time[-1]:.2f}s / {sim_time[-1]:.2f}s (完成)')
        
        actual_duration = time.time() - start_real_time
        expected_duration = (sim_time[-1] - sim_time[0]) / speed_factor
        
        print(f"\n回放结束!")
        print(f"  仿真时长: {sim_time[-1]:.2f}s")
        print(f"  预期播放时长: {expected_duration:.2f}s")
        print(f"  实际播放时长: {actual_duration:.2f}s")
        print(f"  时间误差: {abs(actual_duration - expected_duration):.3f}s")
        
        plt.ioff()
        plt.show()

    def _replay_lightweight(self, results, realtime=True, speed_factor=1.0):
        """
        轻量级回放模式 - 不渲染机器人模型，只显示轨迹点
        速度比完整模式快 10-50 倍
        """
        print("\n正在初始化轻量级回放环境...")

        # 1. 创建简单的 3D 图形
        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 设置视角
        ax.set_xlim([-0.2, 0.6])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([0.0, 0.6])
        ax.view_init(elev=20, azim=-45)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Visual Servoing Simulation (Lightweight Mode)')

        # 2. 提取数据
        target_pos = np.array(results['target_pos'])
        robot_pos = np.array(results['robot_pos'])
        q_traj = results['q']
        sim_time = results['time']
        error_data = results['error']
        total_frames = len(q_traj)

        # 3. 绘制静态轨迹
        ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
                'r--', linewidth=1, alpha=0.3, label='Target Path')

        # 4. 动态元素
        target_marker, = ax.plot([target_pos[0, 0]], [target_pos[0, 1]], [target_pos[0, 2]],
                                  'ro', markersize=10, label='Target')
        robot_marker, = ax.plot([robot_pos[0, 0]], [robot_pos[0, 1]], [robot_pos[0, 2]],
                                 'go', markersize=10, label='Robot EE')
        robot_trail, = ax.plot([], [], [], 'g-', linewidth=2, alpha=0.7, label='Robot Trail')

        # 文本显示
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        error_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        speed_text = ax.text2D(0.02, 0.85, f'Speed: {speed_factor}x (Lightweight)', transform=ax.transAxes, fontsize=10)

        ax.legend(loc='upper right')

        plt.show(block=False)
        plt.pause(0.1)

        print(f"开始回放... (总帧数: {total_frames}, 仿真时长: {sim_time[-1]:.2f}s)")
        print(f"播放速度: {speed_factor}x, 实时模式: {realtime}, 轻量级模式: True")

        # 存储轨迹
        trail_x, trail_y, trail_z = [], [], []

        start_real_time = time.time()
        start_sim_time = sim_time[0]
        target_fps = 60  # 目标帧率
        frame_interval = 1.0 / target_fps

        try:
            frame_idx = 0
            last_update_time = start_real_time

            while frame_idx < total_frames:
                current_real_time = time.time()

                if realtime:
                    # 按真实时间计算当前应该显示的仿真时间
                    elapsed_real_time = current_real_time - start_real_time
                    target_sim_time = start_sim_time + elapsed_real_time * speed_factor

                    # 找到对应的帧索引
                    while frame_idx < total_frames - 1 and sim_time[frame_idx] < target_sim_time:
                        frame_idx += 1
                else:
                    # 非实时模式：固定跳帧
                    frame_idx += max(1, int(total_frames / 500))

                if frame_idx >= total_frames:
                    break

                # 控制刷新率
                if current_real_time - last_update_time < frame_interval:
                    time.sleep(0.001)
                    continue
                last_update_time = current_real_time

                # 更新目标点
                tp = target_pos[frame_idx]
                target_marker.set_data([tp[0]], [tp[1]])
                target_marker.set_3d_properties([tp[2]])

                # 更新机器人末端位置
                rp = robot_pos[frame_idx]
                robot_marker.set_data([rp[0]], [rp[1]])
                robot_marker.set_3d_properties([rp[2]])

                # 更新轨迹 (每隔几帧添加一个点，避免太密集)
                if len(trail_x) == 0 or frame_idx % 5 == 0:
                    trail_x.append(rp[0])
                    trail_y.append(rp[1])
                    trail_z.append(rp[2])
                    robot_trail.set_data(trail_x, trail_y)
                    robot_trail.set_3d_properties(trail_z)

                # 更新文本
                current_sim_time = sim_time[frame_idx]
                current_error = error_data[frame_idx] * 1000
                time_text.set_text(f'Time: {current_sim_time:.2f}s / {sim_time[-1]:.2f}s')
                error_text.set_text(f'Error: {current_error:.2f} mm')

                # 刷新显示 (轻量级，不需要 env.step)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        except KeyboardInterrupt:
            print("\n用户中断")

        # 显示最终状态
        tp = target_pos[-1]
        rp = robot_pos[-1]
        target_marker.set_data([tp[0]], [tp[1]])
        target_marker.set_3d_properties([tp[2]])
        robot_marker.set_data([rp[0]], [rp[1]])
        robot_marker.set_3d_properties([rp[2]])
        time_text.set_text(f'Time: {sim_time[-1]:.2f}s / {sim_time[-1]:.2f}s (完成)')

        actual_duration = time.time() - start_real_time
        expected_duration = (sim_time[-1] - sim_time[0]) / speed_factor

        print(f"\n回放结束!")
        print(f"  仿真时长: {sim_time[-1]:.2f}s")
        print(f"  预期播放时长: {expected_duration:.2f}s")
        print(f"  实际播放时长: {actual_duration:.2f}s")
        print(f"  时间误差: {abs(actual_duration - expected_duration):.3f}s")

        plt.ioff()
        plt.show()

    def _save_animation(self, ani, filename, fps):
        """
        保存动画到文件
        
        :param ani: matplotlib 动画对象
        :param filename: 输出文件名
        :param fps: 帧率
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 根据文件扩展名选择编码器
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.gif':
            writer = animation.PillowWriter(fps=fps)
        elif ext == '.mp4':
            writer = animation.FFMpegWriter(fps=fps, bitrate=3000, 
                                            extra_args=['-vcodec', 'libx264'])
        elif ext == '.avi':
            writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', bitrate=3000)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        
        try:
            ani.save(filename, writer=writer)
            print(f"\n视频已保存: {filename}")
        except Exception as e:
            print(f"\n保存失败: {e}")
            print("提示: 保存 MP4 需要安装 FFmpeg")
            print("  - Windows: 下载 FFmpeg 并添加到 PATH")
            print("  - 或者保存为 GIF 格式")
            
            # 尝试保存为 GIF
            if ext != '.gif':
                gif_filename = os.path.splitext(filename)[0] + '.gif'
                try:
                    gif_writer = animation.PillowWriter(fps=min(fps, 20))
                    ani.save(gif_filename, writer=gif_writer)
                    print(f"已自动保存为 GIF: {gif_filename}")
                except Exception as e2:
                    print(f"GIF 保存也失败: {e2}")


# 便捷函数
def replay(results, realtime=True, speed_factor=1.0):
    """便捷函数：回放仿真结果"""
    saver = VideoSaver()
    saver.replay_simulation(results, realtime=realtime, speed_factor=speed_factor)


def save_video(results, filename='output/simulation.mp4', fps=30, speed_factor=1.0, with_robot=False):
    """
    便捷函数：保存仿真视频
    
    :param results: 仿真结果
    :param filename: 输出文件名
    :param fps: 帧率
    :param speed_factor: 速度倍率
    :param with_robot: 是否包含机器人模型
    """
    saver = VideoSaver()
    if with_robot:
        saver.save_simulation_video_with_robot(results, filename, fps, speed_factor)
    else:
        saver.save_simulation_video(results, filename, fps, speed_factor)