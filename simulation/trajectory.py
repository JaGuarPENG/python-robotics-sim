# trajectory.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class TrajectoryGenerator:
    def __init__(self, mode, init_pos, center, **kwargs):
        """
        轨迹生成器
        
        :param mode: 轨迹模式 - 'circle', 'linear', 'csv'
        :param init_pos: 初始位置 (用于 linear 模式)
        :param center: 中心位置 (用于 circle 模式)
        :param kwargs: 额外参数
            - radius: 圆轨迹半径
            - omega: 圆轨迹角速度
            - velocity: 线性轨迹速度
            - csv_path: CSV 文件路径 (用于 csv 模式)
            - pixel_to_meter: 像素到米的转换系数 (默认 0.001)
            - z_height: CSV 轨迹的 Z 高度 (默认 0.2)
            - image_center: 图像中心像素坐标 [cx, cy] (用于坐标转换)
            - world_offset: 世界坐标偏移 [x, y] (默认 [0, 0])
        """
        self.mode = mode
        self.center = np.array(center)
        self.init_pos = np.array(init_pos)
        self.t = 0
        self.radius = kwargs.get('radius', 0.15)
        self.omega = kwargs.get('omega', 0.5)
        self.velocity = kwargs.get('velocity', 0.01)
        self.duration = None  # 轨迹时长
        
        # CSV 模式参数
        if mode == 'csv':
            csv_path = kwargs.get('csv_path', 'csv/marker_coordinates.csv')
            self.z_height = kwargs.get('z_height', 0.2)
            self.world_offset = np.array(kwargs.get('world_offset', [0.2, 0.0]))  # 世界坐标系偏移
            self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        """加载 CSV 文件并创建插值函数"""
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)
        
        # 保存原始数据用于调试
        self.raw_df = df
        
        # 提取数据列
        self.timestamps = df['时间戳(秒)'].values
        pixel_x = df['X坐标滤波(mm)'].values
        pixel_y = df['Y坐标滤波(mm)'].values
        
        # 保存原始像素数据
        self.raw_pixel_x = pixel_x
        self.raw_pixel_y = pixel_y
        
        # 像素坐标转换为世界坐标 (米)
        # 以图像中心为原点，X 向右为正，Y 向下为正（图像坐标系）
        # 转换为世界坐标系：X 向前，Y 向左
        world_x = pixel_x / 1000 + self.world_offset[0]
        world_y = -pixel_y / 1000 + self.world_offset[1]
        
        self.world_x = world_x
        self.world_y = world_y
        
        # 创建插值函数 (线性插值)
        self.interp_x = interp1d(self.timestamps, world_x, kind='linear', 
                                  bounds_error=False, fill_value=(world_x[0], world_x[-1]))
        self.interp_y = interp1d(self.timestamps, world_y, kind='linear',
                                  bounds_error=False, fill_value=(world_y[0], world_y[-1]))
        
        # 记录轨迹时长
        self.duration = self.timestamps[-1] - self.timestamps[0]
        print(f"[CSV轨迹] 已加载 {len(self.timestamps)} 个点, 时长: {self.duration:.2f}s")
        print(f"[CSV轨迹] X范围: [{world_x.min():.3f}, {world_x.max():.3f}] m")
        print(f"[CSV轨迹] Y范围: [{world_y.min():.3f}, {world_y.max():.3f}] m")

    def get_duration(self):
        """
        获取轨迹的总时长
        
        :return: 时长 (秒)，对于无限轨迹返回 None
        """
        if self.mode == 'csv':
            return self.duration
        else:
            # circle 和 linear 模式理论上是无限的，返回 None
            return None
    
    def is_finished(self):
        """
        检查轨迹是否已结束
        
        :return: True 如果轨迹已结束，否则 False
        """
        if self.mode == 'csv':
            return self.t >= self.duration
        else:
            # circle 和 linear 模式永不结束
            return False

    def update(self, dt):
        self.t += dt

    def get_state(self):
        """返回当前真实的三维位置"""
        if self.mode == 'circle':
            x = self.center[0] + self.radius * np.cos(self.omega * self.t)
            y = self.center[1] + self.radius * np.sin(self.omega * self.t)
            z = self.center[2]
            return np.array([x, y, z])
        
        elif self.mode == 'linear':
            direction = np.array([1.0, 0.0, 0.0])  # 默认沿 X 轴方向
            return self.init_pos + direction * self.velocity * self.t
        
        elif self.mode == 'csv':
            # 使用插值获取当前位置
            x = float(self.interp_x(self.t))
            y = float(self.interp_y(self.t))
            z = self.z_height
            return np.array([x, y, z])
        
        return self.init_pos
    
    def get_trajectory_info(self):
        """获取轨迹信息 (用于调试)"""
        if self.mode == 'csv':
            return {
                'mode': 'csv',
                'duration': self.duration,
                'num_points': len(self.timestamps),
                'x_range': (self.world_x.min(), self.world_x.max()),
                'y_range': (self.world_y.min(), self.world_y.max()),
                'z_height': self.z_height
            }
        elif self.mode == 'circle':
            return {
                'mode': 'circle',
                'center': self.center,
                'radius': self.radius,
                'omega': self.omega,
                'duration': None  # 无限
            }
        elif self.mode == 'linear':
            return {
                'mode': 'linear',
                'init_pos': self.init_pos,
                'velocity': self.velocity,
                'duration': None  # 无限
            }
        return {}
    
    def print_csv_data(self, num_rows=10):
        """
        打印 CSV 数据用于调试
        
        :param num_rows: 打印的行数 (默认前10行，设为 -1 打印全部)
        """
        if self.mode != 'csv':
            print("[警告] 当前不是 CSV 模式，无法打印 CSV 数据")
            return
        
        print("\n" + "=" * 60)
        print("CSV 轨迹数据调试信息")
        print("=" * 60)
        
        # 打印基本信息
        print(f"\n【基本信息】")
        print(f"  总数据点数: {len(self.timestamps)}")
        print(f"  时间范围: [{self.timestamps[0]:.3f}, {self.timestamps[-1]:.3f}] s")
        print(f"  轨迹时长: {self.duration:.3f} s")
        print(f"  Z 高度: {self.z_height} m")
        print(f"  世界坐标偏移: {self.world_offset}")
        
        # 打印原始数据范围
        print(f"\n【原始数据范围 (mm)】")
        print(f"  X: [{self.raw_pixel_x.min():.2f}, {self.raw_pixel_x.max():.2f}]")
        print(f"  Y: [{self.raw_pixel_y.min():.2f}, {self.raw_pixel_y.max():.2f}]")
        
        # 打印转换后数据范围
        print(f"\n【世界坐标范围 (m)】")
        print(f"  X: [{self.world_x.min():.4f}, {self.world_x.max():.4f}]")
        print(f"  Y: [{self.world_y.min():.4f}, {self.world_y.max():.4f}]")
        
        # 打印数据表格
        if num_rows == -1:
            num_rows = len(self.timestamps)
        num_rows = min(num_rows, len(self.timestamps))
        
        print(f"\n【前 {num_rows} 行数据】")
        print("-" * 70)
        print(f"{'索引':>5} | {'时间戳(s)':>10} | {'原始X(mm)':>10} | {'原始Y(mm)':>10} | {'世界X(m)':>10} | {'世界Y(m)':>10}")
        print("-" * 70)
        
        for i in range(num_rows):
            print(f"{i:>5} | {self.timestamps[i]:>10.3f} | {self.raw_pixel_x[i]:>10.2f} | {self.raw_pixel_y[i]:>10.2f} | {self.world_x[i]:>10.4f} | {self.world_y[i]:>10.4f}")
        
        # 如果数据超过显示行数，显示最后几行
        if len(self.timestamps) > num_rows:
            print("...")
            print(f"\n【最后 3 行数据】")
            print("-" * 70)
            for i in range(-3, 0):
                idx = len(self.timestamps) + i
                print(f"{idx:>5} | {self.timestamps[idx]:>10.3f} | {self.raw_pixel_x[idx]:>10.2f} | {self.raw_pixel_y[idx]:>10.2f} | {self.world_x[idx]:>10.4f} | {self.world_y[idx]:>10.4f}")
        
        print("=" * 60 + "\n")
    
    def print_sample_states(self, time_points=None):
        """
        打印指定时间点的状态
        
        :param time_points: 时间点列表，默认为 [0, 1, 2, ..., duration]
        """
        if time_points is None:
            # 默认每秒采样一次
            if self.duration:
                time_points = np.arange(0, self.duration + 0.5, 1.0)
            else:
                time_points = np.arange(0, 10.5, 1.0)
        
        print("\n【指定时间点的状态】")
        print("-" * 50)
        print(f"{'时间(s)':>8} | {'X(m)':>10} | {'Y(m)':>10} | {'Z(m)':>10}")
        print("-" * 50)
        
        original_t = self.t
        for t in time_points:
            self.t = t
            state = self.get_state()
            print(f"{t:>8.2f} | {state[0]:>10.4f} | {state[1]:>10.4f} | {state[2]:>10.4f}")
        self.t = original_t
        
        print("-" * 50 + "\n")


if __name__ == "__main__":
    import config
    
    # 创建 CSV 轨迹生成器
    traj = TrajectoryGenerator(
        mode='csv',
        init_pos=config.TRAJ_INIT_POS,
        center=config.TRAJ_CENTER,
        csv_path=config.TRAJ_CSV_PATH,
        z_height=config.TRAJ_Z_HEIGHT,
        world_offset=config.TRAJ_WORLD_OFFSET
    )
    
    # 打印 CSV 数据 (前 15 行)
    traj.print_csv_data(num_rows=15)
    
    # 打印指定时间点的状态
    traj.print_sample_states()
    
    # 打印轨迹信息
    print("轨迹信息:", traj.get_trajectory_info())