import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import time
import os
from simulation import run_simulation
import config
from tools.video_saver import VideoSaver

def plot_results(results):
    time_data = results['time']
    error = results['error']
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, np.array(error) * 1000, label='Position Error (mm)', color='b')
    plt.title(f"Tracking Error (KP={config.KP}, KI={config.KI})")
    plt.xlabel('Time (s)')
    plt.ylabel('Error (mm)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # 运行仿真
    results = run_simulation(use_real_vision=config.VISION_MODE)
    
    if len(results['time']) > 0:
        # 绘制曲线
        plot_results(results)
        
        # 回放视频
        vid = VideoSaver()
        # vid.save_simulation_video(results, filename='simulation_output.mp4', fps=30, speed_factor=1.0)
        vid.replay_simulation(results, realtime=True, speed_factor=1.0)

    else:
        print("没有采集到足够的数据，无法绘图/回放。")