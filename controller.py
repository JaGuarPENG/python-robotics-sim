# controller.py
import numpy as np

class SimplePBVS:
    def __init__(self, kp, ki, max_step, integral_limit):
        self.kp = kp
        self.ki = ki
        self.max_step = max_step
        self.integral_limit = integral_limit
        self.integral_error = np.zeros(3)

    def compute_next_pose(self, current_pos, desired_target_pos, feedforward_vel, dt):
        # 1. 计算误差
        error = desired_target_pos - current_pos
        
        # 2. 积分累加
        self.integral_error += error * dt
        
        # 3. 积分抗饱和 (Anti-Windup)
        norm_i = np.linalg.norm(self.integral_error)
        if norm_i > self.integral_limit:
            self.integral_error = (self.integral_error / norm_i) * self.integral_limit

        # 4. 控制律: V_cmd = Kp*E + Ki*∫E + V_ff
        v_cmd = (self.kp * error) + \
                (self.ki * self.integral_error) + \
                feedforward_vel
        
        # 5. 计算位移并限幅
        delta_pos = v_cmd * dt
        step_distance = np.linalg.norm(delta_pos)
        if step_distance > self.max_step:
            delta_pos = (delta_pos / step_distance) * self.max_step
            
        return current_pos + delta_pos