"""
将你现有的追踪系统与工控机工艺包集成

思路:
1. 保留现有的视觉检测和卡尔曼滤波
2. 但不再直接控制机械臂
3. 而是通过 TCP 把目标位置发给工控机的工艺包

对比:
- 原方案: 视觉 -> 你的追踪算法 -> 直接发关节角给机器人 (125Hz)
- 新方案: 视觉 -> 发送目标位置 -> 工控机工艺包 (con_fast_catch) -> 机器人
"""

import time
import threading
import numpy as np
from conveyor_package_sender import ConveyorPackageSender


class ConveyorPackageIntegration:
    """
    工艺包集成适配器
    
    替换原有的 ConveyorTrackingService
    改为发送目标位置给工控机
    """
    
    def __init__(self, robot_ip="192.168.0.10", port=9527):
        # 连接到工控机工艺包
        self.sender = ConveyorPackageSender(robot_ip, port)
        
        # 运行状态
        self.is_running = False
        self.vision_thread = None
        
        # 视觉检测接口 (你需要替换为实际的检测器)
        self.vision_detector = None
        
        # 发送频率控制
        self.send_hz = 10.0  # 10Hz发送给工艺包
        self.send_interval = 1.0 / self.send_hz
        
        # 状态机
        self.state = "OBSERVING"  # OBSERVING, TRACKING
        self.last_target_id = None
        
        # 传送带速度 (用于预测)
        self.conveyor_speed_y = 0.05  # m/s
        
        print("[PackageIntegration] 初始化完成")
    
    def set_vision_detector(self, detector):
        """设置视觉检测器"""
        self.vision_detector = detector
    
    def connect(self) -> bool:
        """连接到工控机"""
        return self.sender.connect()
    
    def start(self):
        """启动服务"""
        if self.is_running:
            return
        
        if not self.sender.is_connected:
            print("[错误] 未连接到工控机")
            return
        
        self.is_running = True
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        
        print("[PackageIntegration] 服务已启动")
        print(f"  - 视觉检测频率: 60Hz")
        print(f"  - 数据发送频率: {self.send_hz}Hz")
    
    def stop(self):
        """停止服务"""
        self.is_running = False
        if self.vision_thread:
            self.vision_thread.join(timeout=1.0)
        self.sender.disconnect()
        print("[PackageIntegration] 服务已停止")
    
    def _vision_loop(self):
        """
        视觉检测循环
        
        60Hz运行检测，按 send_hz 频率发送数据给工控机
        """
        last_send_time = 0
        frame_count = 0
        
        while self.is_running:
            loop_start = time.perf_counter()
            
            # 1. 获取检测结果
            target_pos = self._detect_target()
            
            if target_pos is not None:
                # 2. 状态机处理
                if self.state == "OBSERVING":
                    self.state = "TRACKING"
                    print(f"[状态机] 检测到目标，开始追踪")
                
                # 3. 按设定频率发送给工控机
                current_time = time.perf_counter()
                if current_time - last_send_time >= self.send_interval:
                    # 坐标转换: 米 -> 毫米
                    x_mm = target_pos[0] * 1000
                    y_mm = target_pos[1] * 1000
                    z_mm = target_pos[2] * 1000 if len(target_pos) > 2 else None
                    
                    # 发送给工艺包
                    self.sender.send_target(x_mm, y_mm, z_mm, queue_id=0)
                    
                    last_send_time = current_time
                    frame_count += 1
            else:
                # 目标丢失
                if self.state == "TRACKING":
                    self.state = "OBSERVING"
                    print(f"[状态机] 目标丢失，返回观察状态")
            
            # 频率控制 (约60Hz)
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 1/60 - elapsed)
            time.sleep(sleep_time)
    
    def _detect_target(self):
        """
        调用视觉检测器获取目标位置
        
        返回: [x, y, z] 单位: 米 (机器人坐标系)
        或 None 表示未检测到
        """
        if self.vision_detector is None:
            return None
        
        # 这里调用你的视觉检测接口
        # 示例:
        # pos, _ = self.vision_detector.detect()
        # return pos
        
        return None
    
    def set_conveyor_speed(self, speed_m_s: float):
        """设置传送带速度 (用于工控机预测)"""
        self.conveyor_speed_y = speed_m_s
        self.sender.encoder_speed = speed_m_s * 1000 / 100  # 转换为编码器增量


# ============== 使用示例 ==============

def example_usage():
    """
    如何在你的代码中使用
    """
    
    # 1. 创建集成器
    integration = ConveyorPackageIntegration("192.168.0.10", 9527)
    
    # 2. 连接工控机
    if not integration.connect():
        print("连接失败")
        return
    
    # 3. 设置你的视觉检测器
    # integration.set_vision_detector(your_vision_detector)
    
    # 4. 设置传送带速度 (重要！工控机需要这个进行预测)
    integration.set_conveyor_speed(0.05)  # 0.05 m/s
    
    # 5. 启动服务
    integration.start()
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("停止...")
    finally:
        integration.stop()


if __name__ == "__main__":
    example_usage()
