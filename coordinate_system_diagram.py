"""
三维追踪式扫描仪 + 六轴机械臂 坐标系示意图
展示各坐标系之间的关系和转换
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_coordinate_frame_3d(ax, origin, R, scale=1.0, label="", colors=('r', 'g', 'b'), linewidth=2):
    """
    在3D图中绘制坐标系
    R: 3x3旋转矩阵，列向量分别表示x, y, z轴在世界坐标系中的方向
    """
    x_axis = origin + scale * R[:, 0]
    y_axis = origin + scale * R[:, 1]
    z_axis = origin + scale * R[:, 2]
    
    # 绘制坐标轴
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
            color=colors[0], linewidth=linewidth)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
            color=colors[1], linewidth=linewidth)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
            color=colors[2], linewidth=linewidth)
    
    # 添加标签
    offset = 0.15 * scale
    ax.text(x_axis[0] + offset, x_axis[1] + offset, x_axis[2] + offset, 'X', 
            color=colors[0], fontsize=10, fontweight='bold')
    ax.text(y_axis[0] + offset, y_axis[1] + offset, y_axis[2] + offset, 'Y', 
            color=colors[1], fontsize=10, fontweight='bold')
    ax.text(z_axis[0] + offset, z_axis[1] + offset, z_axis[2] + offset, 'Z', 
            color=colors[2], fontsize=10, fontweight='bold')
    
    # 坐标系名称
    if label:
        ax.text(origin[0], origin[1], origin[2] - 0.5*scale, label, 
                fontsize=11, fontweight='bold', ha='center', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

def draw_robot_arm_simple(ax, base_pos, end_pos, color='steelblue'):
    """简化绘制机械臂"""
    # 绘制从基座到末端的线表示机械臂
    ax.plot([base_pos[0], end_pos[0]], [base_pos[1], end_pos[1]], [base_pos[2], end_pos[2]], 
            color=color, linewidth=6, alpha=0.7)
    
    # 绘制关节点
    ax.scatter(*base_pos, color='darkblue', s=100, marker='s')
    ax.scatter(*end_pos, color='red', s=100, marker='o')

def draw_scanner(ax, pos, R, scale=0.3):
    """绘制扫描仪示意"""
    # 扫描仪主体（长方体示意）
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # 简化为一个指向Z方向的锥形表示扫描方向
    tip = pos + scale * 1.5 * R[:, 2]
    
    # 绘制扫描光束
    ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]], 
            'c--', linewidth=2, alpha=0.6)
    
    # 扫描仪位置
    ax.scatter(*pos, color='cyan', s=80, marker='^')

def draw_tracked_object(ax, pos, R, scale=0.4):
    """绘制被扫描的物体"""
    # 物体坐标系
    draw_coordinate_frame_3d(ax, pos, R, scale=scale, label="{Object}", 
                             colors=('orange', 'purple', 'brown'))
    
    # 物体实体（立方体示意）
    ax.scatter(*pos, color='orange', s=150, marker='D', alpha=0.5)

# ==================== 主程序 ====================

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 定义各个坐标系的位置和姿态
# 追踪仪坐标系（世界参考系）
T_tracker = np.array([0, 0, 0])
R_tracker = np.eye(3)  # 追踪仪作为参考

# 机械臂基座坐标系
T_base = np.array([1.5, 0, 0])
R_base = np.eye(3)

# 机械臂末端坐标系（假设某个姿态）
T_end = np.array([1.5, 0.5, 1.2])
# 末端稍微倾斜
angle = np.pi/6
R_end = np.array([
    [np.cos(angle), 0, np.sin(angle)],
    [0, 1, 0],
    [-np.sin(angle), 0, np.cos(angle)]
])

# 扫描仪坐标系（安装在末端上）
# 扫描仪相对于末端有一定偏移和旋转
T_scanner = T_end + np.array([0.1, 0, 0.15])
R_scanner = R_end @ np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])  # 扫描仪可能有安装角度

# 工具坐标系（在末端上）
T_tool = T_end + np.array([0, 0, 0.2])
R_tool = R_end

# 物体坐标系（被扫描的对象）
T_object = np.array([2.0, 1.0, 0.8])
R_object = np.eye(3)

# ==================== 绘制 ====================

# 1. 追踪仪坐标系（参考坐标系）
draw_coordinate_frame_3d(ax, T_tracker, R_tracker, scale=0.6, 
                         label="{Tracker}", colors=('r', 'g', 'b'), linewidth=3)
ax.text(T_tracker[0], T_tracker[1], T_tracker[2] + 0.8, 
        '(World Reference)', fontsize=9, ha='center', style='italic', color='gray')

# 2. 机械臂基座
draw_coordinate_frame_3d(ax, T_base, R_base, scale=0.4, 
                         label="{Robot Base}", colors=('darkred', 'darkgreen', 'darkblue'))

# 3. 绘制机械臂连杆
draw_robot_arm_simple(ax, T_base, T_end)

# 4. 机械臂末端
draw_coordinate_frame_3d(ax, T_end, R_end, scale=0.35, 
                         label="{End-effector}", colors=('darkred', 'darkgreen', 'darkblue'))

# 5. 扫描仪
draw_coordinate_frame_3d(ax, T_scanner, R_scanner, scale=0.3, 
                         label="{Scanner}", colors=('c', 'm', 'y'))
draw_scanner(ax, T_scanner, R_scanner, scale=0.4)

# 6. 工具坐标系
draw_coordinate_frame_3d(ax, T_tool, R_tool, scale=0.25, 
                         label="{Tool}", colors=('darkcyan', 'darkmagenta', 'olive'))

# 7. 被扫描物体
draw_tracked_object(ax, T_object, R_object, scale=0.4)

# ==================== 绘制变换关系连线 ====================

# 绘制扫描光束（从扫描仪指向物体）
ax.plot([T_scanner[0], T_object[0]], [T_scanner[1], T_object[1]], [T_scanner[2], T_object[2]], 
        'c:', linewidth=2, alpha=0.5)

# ==================== 图例和说明 ====================

# 添加图例说明
legend_text = """
坐标系说明:
┌─────────────────────────────────────────────────────────┐
│  {Tracker}      追踪仪坐标系 (世界参考系)                │
│  {Robot Base}   机械臂基座坐标系                         │
│  {End-effector} 机械臂末端坐标系                         │
│  {Scanner}      扫描仪坐标系                             │
│  {Tool}         工具坐标系                               │
│  {Object}       被扫描物体坐标系                         │
├─────────────────────────────────────────────────────────┤
│  变换关系:                                               │
│  T_tracker^base   : 追踪仪 → 机械臂基座 (手眼标定)       │
│  T_base^end       : 机械臂基座 → 末端 (正运动学)         │
│  T_end^scanner    : 末端 → 扫描仪 (安装标定)             │
│  T_end^tool       : 末端 → 工具 (TCP标定)                │
│  T_scanner^object : 扫描仪 → 物体 (扫描测量)             │
└─────────────────────────────────────────────────────────┘
"""

fig.text(0.02, 0.02, legend_text, fontsize=10, family='monospace',
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 设置坐标轴
ax.set_xlabel('X (m)', fontsize=11)
ax.set_ylabel('Y (m)', fontsize=11)
ax.set_zlabel('Z (m)', fontsize=11)
ax.set_title('三维追踪扫描系统 - 坐标系关系示意图', fontsize=14, fontweight='bold', pad=20)

# 设置视角
ax.view_init(elev=20, azim=-60)

# 设置坐标范围
ax.set_xlim(-0.5, 3)
ax.set_ylim(-0.5, 2)
ax.set_zlim(-0.2, 2)

plt.tight_layout()
plt.savefig('coordinate_systems_diagram.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
print("示意图已保存为: coordinate_systems_diagram.png")
