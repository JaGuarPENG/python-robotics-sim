"""
机器人运动安全卫士
"""

import time
import numpy as np

class SafetyGuard:
    """负责运动指令的拦截与合法性检查"""

    def __init__(self, max_vel_ms: float = 0.5, max_inc_mm: float = 50.0):
        self.max_vel_ms = max_vel_ms
        self.max_inc_mm = max_inc_mm
        self.last_send_time = 0.0

    def check_increment(self, dx, dy, dz) -> (bool, str):
        """检查单次增量是否过大"""
        if abs(dx) > self.max_inc_mm or abs(dy) > self.max_inc_mm or abs(dz) > self.max_inc_mm:
            return False, f"单次增量超过 {self.max_inc_mm}mm 安全阈值"
        return True, ""

    def check_velocity(self, dx_mm, dy_mm, dz_mm) -> (bool, str):
        """检查运动速度是否过快"""
        now = time.time()
        dt = now - self.last_send_time
        
        if self.last_send_time == 0:
            self.last_send_time = now
            return True, ""

        if dt < 0.001: # 防止除零
            return True, ""

        dist_m = np.sqrt((dx_mm/1000.0)**2 + (dy_mm/1000.0)**2 + (dz_mm/1000.0)**2)
        velocity = dist_m / dt
        
        if velocity > self.max_vel_ms:
            return False, f"指令速度过快 ({velocity:.2f} m/s > {self.max_vel_ms} m/s)"
        
        self.last_send_time = now
        return True, ""
