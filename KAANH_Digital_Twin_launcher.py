"""
机器人数字孪生仿真平台 GUI - 启动脚本

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
    python KAANH_Digital_Twin_launcher.py
    或
    python -m KAANH_Digital_Twin.app
"""

from KAANH_Digital_Twin import main

if __name__ == "__main__":
    main()
