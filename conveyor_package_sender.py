"""
佳安传送带工艺包 - 简化版发送器
可以直接集成到你的现有代码中

使用示例:
    sender = ConveyorPackageSender("192.168.0.10", 9527)
    sender.connect()
    
    # 当检测到目标时
    sender.send_target(x_mm, y_mm)
"""

import socket
import time
import threading
from typing import Optional


class ConveyorPackageSender:
    """
    传送带工艺包数据发送器
    
    协议: camera_data --POS=encoder --type=queue --x=x_mm --y=y_mm
    """
    
    def __init__(self, robot_ip: str = "192.168.0.10", port: int = 9527):
        self.robot_ip = robot_ip
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.is_connected = False
        
        # 传送带编码器模拟
        self._encoder_value = 0.0
        self._encoder_lock = threading.Lock()
        self._encoder_thread = None
        self._running = False
        
        # 编码器递增速度 (模拟传送带运动)
        self.encoder_speed = 100.0  # 每10ms增加的脉冲数
    
    def connect(self) -> bool:
        """连接到工控机"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.robot_ip, self.port))
            self.is_connected = True
            
            # 启动编码器更新线程
            self._start_encoder_simulation()
            
            print(f"[ConveyorSender] 已连接到 {self.robot_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[ConveyorSender] 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        if self._encoder_thread:
            self._encoder_thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
        self.is_connected = False
        print("[ConveyorSender] 已断开")
    
    def _start_encoder_simulation(self):
        """启动编码器模拟线程"""
        self._running = True
        self._encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self._encoder_thread.start()
    
    def _encoder_loop(self):
        """模拟传送带编码器递增"""
        while self._running:
            with self._encoder_lock:
                self._encoder_value += self.encoder_speed
            time.sleep(0.01)  # 10ms更新一次
    
    def get_current_encoder(self) -> float:
        """获取当前编码器值"""
        with self._encoder_lock:
            return self._encoder_value
    
    def send_target(self, x_mm: float, y_mm: float, 
                   z_mm: Optional[float] = None,
                   queue_id: int = 0) -> bool:
        """
        发送目标位置给工控机
        
        Args:
            x_mm: 目标X坐标 (毫米，机器人基座标系)
            y_mm: 目标Y坐标 (毫米)
            z_mm: 目标Z坐标 (毫米，可选)
            queue_id: 物料队列号 (多物料时使用)
        
        Returns:
            是否发送成功
        """
        if not self.is_connected or self.socket is None:
            return False
        
        encoder_pos = self.get_current_encoder()
        
        # 构建协议字符串
        if z_mm is not None:
            data = f"camera_data --POS={encoder_pos:.3f} --type={queue_id} --x={x_mm:.6f} --y={y_mm:.6f} --z={z_mm:.6f}"
        else:
            data = f"camera_data --POS={encoder_pos:.3f} --type={queue_id} --x={x_mm:.6f} --y={y_mm:.6f}"
        
        try:
            self.socket.sendall(data.encode('utf-8'))
            print(f"[ConveyorSender] 发送: x={x_mm:.1f}, y={y_mm:.1f}, enc={encoder_pos:.0f}")
            return True
        except Exception as e:
            print(f"[ConveyorSender] 发送失败: {e}")
            self.is_connected = False
            return False
    
    def send_with_conveyor_pos(self, x_mm: float, y_mm: float, 
                                conveyor_pos: float,
                                queue_id: int = 0) -> bool:
        """
        使用指定的传送带位置发送目标
        
        如果你能从工控机获取真实的传送带编码器值，使用此方法
        """
        if not self.is_connected or self.socket is None:
            return False
        
        data = f"camera_data --POS={conveyor_pos:.3f} --type={queue_id} --x={x_mm:.6f} --y={y_mm:.6f}"
        
        try:
            self.socket.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"[ConveyorSender] 发送失败: {e}")
            self.is_connected = False
            return False


# ============== 快速测试 ==============

if __name__ == "__main__":
    print("=" * 50)
    print("传送带工艺包发送器测试")
    print("=" * 50)
    
    sender = ConveyorPackageSender("192.168.0.10", 9527)
    
    if not sender.connect():
        print("连接失败，退出")
        exit(1)
    
    print("\n测试: 模拟发送5个目标点...")
    
    # 模拟5个目标点
    test_targets = [
        (400.0, 100.0),
        (400.0, 150.0),
        (400.0, 200.0),
        (400.0, 250.0),
        (400.0, 300.0),
    ]
    
    for i, (x, y) in enumerate(test_targets):
        print(f"\n--- 目标 {i+1} ---")
        sender.send_target(x, y, queue_id=0)
        time.sleep(1.0)  # 每秒发送一个
    
    print("\n测试完成")
    sender.disconnect()
