# estimator.py
import numpy as np

class KalmanFilterEstimator:
    """
    基于卡尔曼滤波的状态估计器
    状态向量: [x, y, vx, vy, ax, ay] (位置、速度、加速度 - Constant Acceleration 模型)
    """
    def __init__(self, dt, process_noise=0.1, measurement_noise=1.0, vel_cov=200.0, acc_cov=200.0):
        self.dt = dt
        self.initialized = False
        
        # 状态向量: [x, y, vx, vy, ax, ay]
        self.state_dim = 6
        self.meas_dim = 2
        
        # 状态转移矩阵 F (x = x + vt + 0.5at^2)
        self.F = np.eye(6)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = 0.5 * dt**2
        self.F[1, 5] = 0.5 * dt**2
        self.F[2, 4] = dt
        self.F[3, 5] = dt
        
        # 测量矩阵 H (只测量位置 x, y)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        
        # 过程噪声 Q
        q_sigma = process_noise
        self.Q = np.eye(6) * (q_sigma ** 2)
        
        # 测量噪声 R
        self.R = np.eye(2) * (measurement_noise ** 2)
        
        # 初始协方差 P
        self.P = np.eye(6)
        self.P[0,0] = self.P[1,1] = measurement_noise**2  # 位置不确定性 (因为已经用测量初始化了，所以可以小)
        self.P[2,2] = self.P[3,3] = vel_cov # 速度不确定性 (假设初始速度未知，给大点)
        self.P[4,4] = self.P[5,5] = acc_cov  # 加速度不确定性
        # self.P = np.eye(6) * 100
        
        # 初始状态 x
        self.x = np.zeros(6)

    def update(self, measurement):
        """
        输入: measurement [x, y] 或 [x, y, z]
        输出: (vel_est, acc_est)
        """
        z = np.array(measurement[:2])  # 只取 x, y
        
        if not self.initialized:
            self.x[0] = z[0]
            self.x[1] = z[1]
            self.initialized = True
            return np.zeros(3), np.zeros(3)

        # --- 1. 预测 (Predict) ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # --- 2. 更新 (Update) ---
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred
        
        # 提取结果
        vx, vy = self.x[2], self.x[3]
        ax, ay = self.x[4], self.x[5]
        
        vel_est = np.array([vx, vy, 0.0])
        acc_est = np.array([ax, ay, 0.0])
        
        return vel_est, acc_est

    def predict_future(self, total_pred_time):
        """基于当前卡尔曼状态进行外推"""
        F_p = np.eye(6)
        F_p[0, 2] = total_pred_time
        F_p[1, 3] = total_pred_time
        F_p[0, 4] = 0.5 * total_pred_time**2
        F_p[1, 5] = 0.5 * total_pred_time**2
        F_p[2, 4] = total_pred_time
        F_p[3, 5] = total_pred_time
        
        x_future = F_p @ self.x
        
        pred_pos = np.array([x_future[0], x_future[1], 0.0])
        pred_vel = np.array([x_future[2], x_future[3], 0.0])
        
        return pred_pos, pred_vel
    
    def get_state(self):
        """返回当前估计的状态"""
        return {
            'pos': np.array([self.x[0], self.x[1], 0.0]),
            'vel': np.array([self.x[2], self.x[3], 0.0]),
            'acc': np.array([self.x[4], self.x[5], 0.0])
        }


class StateEstimator:
    """简单差分估计器"""
    def __init__(self, dt_vision):
        self.dt = dt_vision
        self.last_pos = None
        self.last_vel = None
        self.curr_vel = np.zeros(3)
        self.curr_acc = np.zeros(3)

    def update(self, current_measured_pos):
        """仅在视觉更新时调用"""
        if self.last_pos is not None:
            self.curr_vel = (current_measured_pos - self.last_pos) / self.dt
        else:
            self.curr_vel = np.zeros(3)

        if self.last_vel is not None:
            self.curr_acc = (self.curr_vel - self.last_vel) / self.dt
        else:
            self.curr_acc = np.zeros(3)

        self.last_pos = current_measured_pos
        self.last_vel = self.curr_vel.copy()
        
        return self.curr_vel, self.curr_acc

    def predict(self, delayed_pos, total_pred_time):
        """二阶外推预测"""
        pred_pos = delayed_pos + \
                   self.curr_vel * total_pred_time + \
                   0.5 * self.curr_acc * (total_pred_time**2)
        
        pred_vel = self.curr_vel + self.curr_acc * total_pred_time
        return pred_pos, pred_vel



if __name__ == "__main__":
    print("测试 KalmanFilterEstimator")