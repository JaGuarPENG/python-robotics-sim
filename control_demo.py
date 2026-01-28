import time
import threading
import queue
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt

# 导入模块 (确保路径正确)
from tools.robot_dh import create_ka_ur
from tools.robot_client import ArisRobotClient

# 全局变量
shared_data = {
    "joints": None,     
    "running": True,    
}
data_lock = threading.Lock() 
cmd_queue = queue.Queue()

ROBOT_IP = "192.168.1.5"
PORT_MONITOR = 5888  # 监听端口
PORT_CONTROL = 5999  # 控制端口

# ==========================================================
# 线程 1: 监控线程 (连 5888) - 只管读，拼命读
# ==========================================================
def monitor_thread_func():
    print(f"[监控线程] 启动，连接 {ROBOT_IP}:{PORT_MONITOR}...")
    # 超时设短 (3s)，保证实时性
    client = ArisRobotClient(ROBOT_IP, PORT_MONITOR, timeout=3)
    
    if not client.connect():
        print("[监控线程] 连接失败 (请检查5888端口)")
        return

    try:
        client.login("Engineer", "000000")
        print("[监控线程] 准备就绪，开始实时刷新...")
        
        while shared_data["running"]:
            # 发送 get，永不阻塞
            data = client.get_status()
            
            # 解析数据
            new_joints = None
            if data and 'ret_context' in data:
                ctx = data['ret_context']
                if 'motion_msg' in ctx:
                    motion = ctx['motion_msg']
                    if 'motor_pos' in motion:
                        m_pos = motion['motor_pos']
                        if isinstance(m_pos, list) and len(m_pos) > 0:
                            new_joints = m_pos[0]

            if new_joints:
                with data_lock:
                    shared_data["joints"] = new_joints
            
            # 50Hz 高频刷新
            time.sleep(0.02)

    except Exception as e:
        print(f"[监控线程] 异常: {e}")
    finally:
        client.close()
        print("[监控线程] 退出")

# ==========================================================
# 线程 2: 控制线程 (连 5999) - 只管发，慢慢等
# ==========================================================
def control_thread_func():
    print(f"[控制线程] 启动，连接 {ROBOT_IP}:{PORT_CONTROL}...")
    # 超时设长 (60s)，允许长动作
    client = ArisRobotClient(ROBOT_IP, PORT_CONTROL, timeout=60)
    
    if not client.connect():
        print("[控制线程] 连接失败 (请检查5999端口是否已开启！)")
        return

    try:
        client.login("Engineer", "000000")
        time.sleep(0.5)
        client.manual_enable()
        print("[控制线程] 准备就绪，等待按键指令...")
        client.set_pgm_vel(100)  # 设置程序速度为 100%
        client.set_jog_vel(100)  # 设置JOG速度为 100%
        print("[控制线程] 速度设置为 100%")

        while shared_data["running"]:
            try:
                # 阻塞等待队列指令
                cmd = cmd_queue.get(timeout=1.0)
                
                if cmd['type'] == 'movej':
                    print(f"\n>>> [执行] MoveJ 开始: {cmd['joints']}")
                    
                    # 这里的阻塞只会卡住这个线程，完全不影响监控线程的画面
                    res = client.movej(cmd['joints'], cmd['vels'])
                    
                    print(f"<<< [结束] MoveJ 完成. 返回: {res}")
                    
                cmd_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[控制线程] 指令错误: {e}")

    except Exception as e:
        print(f"[控制线程] 异常: {e}")
    finally:
        client.close()
        print("[控制线程] 退出")

# ==========================================================
# 键盘交互
# ==========================================================
def on_key_press(event):
    if event.key == '1':
        print("\n[按键] 1 -> 回零位")
        target = [0, 0, 0, 0, 0, 0]
        vel = [100, 200, 100]
        cmd_queue.put({'type': 'movej', 'joints': target, 'vels': vel})
        
    elif event.key == '2':
        print("\n[按键] 2 -> 移动到目标点")
        # 这里的角度根据你的机器人实际情况调整
        target = [0, 0, 150, -60, -90, 0] 
        vel = [100, 200, 100]
        cmd_queue.put({'type': 'movej', 'joints': target, 'vels': vel})

    elif event.key == 'q':
        shared_data["running"] = False

# ==========================================================
# 主程序
# ==========================================================
def main():
    # 1. 启动监控线程 (Port 5888)
    t_mon = threading.Thread(target=monitor_thread_func, daemon=True)
    t_mon.start()

    # 2. 启动控制线程 (Port 5999)
    # 给一点时间让上一个连接建立好
    time.sleep(1) 
    t_ctrl = threading.Thread(target=control_thread_func, daemon=True)
    t_ctrl.start()

    # 3. 初始化 3D 界面
    print("[主界面] 加载 3D 模型...")
    robot = create_ka_ur()
    
    env = rtb.backends.PyPlot.PyPlot()
    env.launch()
    env.add(robot)
    
    env.ax.set_xlim([-0.8, 0.8])
    env.ax.set_ylim([-0.8, 0.8])
    env.ax.set_zlim([0.0, 1.2])

    env.ax.figure.canvas.mpl_connect('key_press_event', on_key_press)

    print("\n" + "="*60)
    print("   双端口并发模式 (5888:Monitor, 5999:Control)")
    print("   [1] : 回零位")
    print("   [2] : 移动到目标点")
    print("   [Q] : 退出")
    print("="*60)

    # 4. 绘图循环
    try:
        while shared_data["running"]:
            if not env.ax.figure.canvas.manager.window: break

            current_joints = None
            with data_lock:
                if shared_data["joints"]:
                    # 假设机器人返回角度 -> 转弧度
                    current_joints = [np.deg2rad(x) for x in shared_data["joints"]]
            
            if current_joints:
                robot.q = np.array(current_joints)
            
            env.step(0.05)
            
    except KeyboardInterrupt:
        pass
    finally:
        shared_data["running"] = False
        t_mon.join(timeout=1)
        t_ctrl.join(timeout=1)
        print("程序结束")

if __name__ == "__main__":
    main()