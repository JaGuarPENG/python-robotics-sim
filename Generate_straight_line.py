import numpy as np
import pandas as pd

def generate_trajectory_data():
    # --- 参数设置 ---
    num_points = 1000
    length_mm = 250.0        # 0.25m = 250mm (为了留在 0.6m 工作空间内)
    fps = 100                # 假设相机/传感器帧率为 100 fps
    noise_std = 2.0          # 原始数据的模拟噪声标准差 (单位: mm)
    marker_id = 1            # 假设标记点的 ID 为 1

    # 1. 帧号: 1 到 1000
    frames = np.arange(1, num_points + 1)

    # 2. 标记ID: 全部为 1
    marker_ids = np.full(num_points, marker_id)

    # 3. 时间戳 (秒): 从 0 开始，每帧增加 1/fps 秒
    timestamps = (frames - 1) / fps

    # 4. 滤波后的坐标 (理想状态的轨迹)
    # X 轴固定在 400mm，Y 轴分布在 -250 到 250 之间 (总长 500mm)，Z 轴保持为 200 (mm)
    x_filtered = np.full(num_points, 400.0)
    y_filtered = np.linspace(-length_mm/2, length_mm/2, num_points)
    z_filtered = np.full(num_points, 200.0) 
    rx = np.full(num_points, 180.0)
    ry = np.zeros(num_points)
    rz = np.zeros(num_points)

    # 5. 原始坐标 (添加高斯噪声以模拟真实环境)
    x_raw = x_filtered + np.random.normal(loc=0.0, scale=noise_std, size=num_points)
    y_raw = y_filtered + np.random.normal(loc=0.0, scale=noise_std, size=num_points)
    z_raw = z_filtered + np.random.normal(loc=0.0, scale=noise_std, size=num_points)

    # --- 构建数据表 ---
    # 必须匹配 VisionPanel 加载的列序: x, y, z, rx, ry, rz
    data = {
        'x': np.round(x_filtered, 3),
        'y': np.round(y_filtered, 3),
        'z': np.round(z_filtered, 3),
        'rx': rx,
        'ry': ry,
        'rz': rz,
        '时间戳(秒)': np.round(timestamps, 3)
    }
    
    df = pd.DataFrame(data)

    # --- 保存为 CSV 文件 ---
    output_filename = 'linear_trajectory_0.5m.csv'
    # 使用 utf-8-sig 编码防止在 Excel 中打开时中文乱码
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 成功生成 {num_points} 个点的数据！")
    print(f"文件已保存至当前目录: {output_filename}")
    print("\n数据前5行预览:")
    print(df.head())

if __name__ == "__main__":
    generate_trajectory_data()