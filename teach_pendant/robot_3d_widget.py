"""
3D 机械臂可视化控件 - 基于 PyVista 的游戏风格渲染
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

import pyvista as pv
from pyvistaqt import QtInteractor

from tools.robot_dh import create_ka_ur


class Robot3DWidget(QWidget):
    """3D 机械臂可视化控件 - PyVista 版本"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置大小策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建 PyVista 渲染器
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # 初始化机器人模型
        self.robot = create_ka_ur()
        self.current_joints = np.zeros(6)

        # 连杆颜色 (金属质感)
        self.link_colors = [
            '#E74C3C',  # 红
            '#F39C12',  # 橙
            '#3498DB',  # 蓝
            '#2ECC71',  # 绿
            '#9B59B6',  # 紫
            '#1ABC9C',  # 青
            '#E74C3C',  # 红 (备用)
        ]

        # 存储预创建的网格和 actor
        self.link_meshes = []
        self.link_actors = []
        self.joint_meshes = []
        self.joint_actors = []
        self.end_effector_mesh = None
        self.end_effector_actor = None
        self.ee_frame_actors = []
        self.tcp_text_actor = None
        self.waypoint_actors = []  # 测试点位标记
        self.trajectory_actors = [] # 轨迹线 actor

        # 测试点位 (X, Y, Z, Label) - 单位: mm
        # 默认为空，由 main_window 设置
        self.test_waypoints = []

        # 设置场景
        self.setup_scene()

        # 预创建机械臂网格
        self._create_robot_meshes()

        # 初次绘制
        self.update_robot(self.current_joints)

    def set_trajectory(self, points_mm):
        """
        设置并绘制轨迹
        Args:
            points_mm: List of (x, y, z) tuples in mm
        """
        self.clear_trajectory()
        
        if not points_mm or len(points_mm) < 2:
            return

        try:
            points_m = np.array(points_mm) / 1000.0
            
            # 使用 PyVista 创建连接线
            lines = pv.lines_from_points(points_m)
            
            actor = self.plotter.add_mesh(
                lines, 
                color='#00FFFF', # 青色
                line_width=4,
                render=True
            )
            self.trajectory_actors.append(actor)
        except Exception as e:
            print(f"绘制轨迹失败: {e}")

    def clear_trajectory(self):
        """清除轨迹显示"""
        for actor in self.trajectory_actors:
            self.plotter.remove_actor(actor)
        self.trajectory_actors = []
        self.plotter.render()

    def setup_scene(self):
        """设置3D场景"""
        # 深色背景 - 渐变色
        self.plotter.set_background('#1a1a2e', top='#16213e')

        # 添加中心坐标轴
        self.add_origin_axes()

        # 添加地面网格
        self.add_ground_grid()

        # 设置相机位置
        self.plotter.camera_position = [
            (1.2, -1.2, 0.8),   # 相机位置
            (0, 0, 0.3),        # 焦点
            (0, 0, 1)           # 上方向
        ]

        # 启用抗锯齿
        self.plotter.enable_anti_aliasing()

    def add_origin_axes(self, length=0.15, radius=0.008):
        """添加原点坐标轴"""
        # X 轴 - 红色
        x_axis = pv.Cylinder(center=(length/2, 0, 0), direction=(1, 0, 0),
                            radius=radius, height=length)
        x_cone = pv.Cone(center=(length + 0.02, 0, 0), direction=(1, 0, 0),
                        height=0.04, radius=0.015)
        self.plotter.add_mesh(x_axis, color='#FF4444', smooth_shading=True)
        self.plotter.add_mesh(x_cone, color='#FF4444', smooth_shading=True)

        # Y 轴 - 绿色
        y_axis = pv.Cylinder(center=(0, length/2, 0), direction=(0, 1, 0),
                            radius=radius, height=length)
        y_cone = pv.Cone(center=(0, length + 0.02, 0), direction=(0, 1, 0),
                        height=0.04, radius=0.015)
        self.plotter.add_mesh(y_axis, color='#44FF44', smooth_shading=True)
        self.plotter.add_mesh(y_cone, color='#44FF44', smooth_shading=True)

        # Z 轴 - 蓝色
        z_axis = pv.Cylinder(center=(0, 0, length/2), direction=(0, 0, 1),
                            radius=radius, height=length)
        z_cone = pv.Cone(center=(0, 0, length + 0.02), direction=(0, 0, 1),
                        height=0.04, radius=0.015)
        self.plotter.add_mesh(z_axis, color='#4444FF', smooth_shading=True)
        self.plotter.add_mesh(z_cone, color='#4444FF', smooth_shading=True)

        # 坐标轴标签
        self.plotter.add_point_labels(
            [(length + 0.05, 0, 0), (0, length + 0.05, 0), (0, 0, length + 0.05)],
            ['X', 'Y', 'Z'],
            font_size=16,
            text_color='white',
            font_family='arial',
            bold=True,
            shape=None,
            always_visible=True
        )

    def add_ground_grid(self):
        """添加地面网格"""
        grid_size = 1.2

        # 创建实体地面平面
        ground = pv.Plane(
            center=(0, 0, -0.001),
            direction=(0, 0, 1),
            i_size=grid_size,
            j_size=grid_size,
            i_resolution=1,
            j_resolution=1
        )
        self.plotter.add_mesh(ground, color='#1a1a2e', opacity=0.8)

        # 主网格线 (粗线，间隔 0.1m)
        main_spacing = 0.1
        main_divisions = int(grid_size / main_spacing)

        for i in range(-main_divisions // 2, main_divisions // 2 + 1):
            pos = i * main_spacing
            # X 方向线
            line_x = pv.Line((-grid_size/2, pos, 0), (grid_size/2, pos, 0))
            self.plotter.add_mesh(line_x, color='#4a5568', line_width=1, opacity=0.6)
            # Y 方向线
            line_y = pv.Line((pos, -grid_size/2, 0), (pos, grid_size/2, 0))
            self.plotter.add_mesh(line_y, color='#4a5568', line_width=1, opacity=0.6)

        # 中心十字线 (更粗更亮)
        center_x = pv.Line((-grid_size/2, 0, 0.001), (grid_size/2, 0, 0.001))
        center_y = pv.Line((0, -grid_size/2, 0.001), (0, grid_size/2, 0.001))
        self.plotter.add_mesh(center_x, color='#718096', line_width=2)
        self.plotter.add_mesh(center_y, color='#718096', line_width=2)

    def _create_robot_meshes(self):
        """预创建机械臂网格对象"""
        # 创建 7 个连杆 (最多7段)
        for i in range(7):
            # 初始创建一个小圆柱体作为占位
            mesh = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                              radius=0.025, height=0.001)
            color = self.link_colors[i % len(self.link_colors)]
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                smooth_shading=True,
                specular=0.5,
                specular_power=15,
                render=False
            )
            self.link_meshes.append(mesh)
            self.link_actors.append(actor)

        # 创建 7 个关节球体
        for i in range(7):
            mesh = pv.Sphere(center=(0, 0, 0), radius=0.035)
            actor = self.plotter.add_mesh(
                mesh,
                color='#ecf0f1',
                smooth_shading=True,
                specular=0.8,
                specular_power=20,
                render=False
            )
            self.joint_meshes.append(mesh)
            self.joint_actors.append(actor)

        # 创建末端执行器
        self.end_effector_mesh = pv.Sphere(center=(0, 0, 0), radius=0.03)
        self.end_effector_actor = self.plotter.add_mesh(
            self.end_effector_mesh,
            color='#00FF00',
            smooth_shading=True,
            specular=1.0,
            specular_power=30,
            render=False
        )

        # 创建末端坐标系的线条
        for i, color in enumerate(['#FF0000', '#00FF00', '#0000FF']):
            line = pv.Line((0, 0, 0), (0.08, 0, 0))
            actor = self.plotter.add_mesh(line, color=color, line_width=3, render=False)
            self.ee_frame_actors.append(actor)

        # 预创建 TCP 文本 (避免每次更新都重建)
        self.tcp_text_actor = self.plotter.add_text(
            'TCP: X=0.0 Y=0.0 Z=0.0 mm',
            position='upper_left',
            font_size=10,
            color='#00FF00',
            font='arial',
            render=False
        )

    def _add_test_waypoints(self):
        """添加测试点位标记"""
        if not self.test_waypoints:
            return

        # 点位颜色 (黄色系，与机器人颜色区分)
        waypoint_colors = ['#FFD700', '#FFA500', '#FF6347']  # 金色, 橙色, 番茄红

        for i, (x, y, z, label) in enumerate(self.test_waypoints):
            # 转换为米
            pos = (x / 1000.0, y / 1000.0, z / 1000.0)
            color = waypoint_colors[i % len(waypoint_colors)]

            # 创建小球体标记
            sphere = pv.Sphere(center=pos, radius=0.015)
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                smooth_shading=True,
                specular=0.8,
                opacity=0.9
            )
            self.waypoint_actors.append(actor)

            # 创建十字标记 (更容易看清位置)
            cross_size = 0.02
            line_x = pv.Line((pos[0] - cross_size, pos[1], pos[2]),
                            (pos[0] + cross_size, pos[1], pos[2]))
            line_y = pv.Line((pos[0], pos[1] - cross_size, pos[2]),
                            (pos[0], pos[1] + cross_size, pos[2]))
            line_z = pv.Line((pos[0], pos[1], pos[2] - cross_size),
                            (pos[0], pos[1], pos[2] + cross_size))
            self.plotter.add_mesh(line_x, color=color, line_width=2)
            self.plotter.add_mesh(line_y, color=color, line_width=2)
            self.plotter.add_mesh(line_z, color=color, line_width=2)

        # 添加点位标签
        label_positions = [(x/1000.0, y/1000.0, z/1000.0 + 0.03)
                          for x, y, z, _ in self.test_waypoints]
        labels = [f"{label}\n({x},{y},{z})"
                 for x, y, z, label in self.test_waypoints]

        self.plotter.add_point_labels(
            label_positions,
            labels,
            font_size=12,
            text_color='#FFD700',
            font_family='arial',
            bold=True,
            shape=None,
            always_visible=True
        )

    def update_robot(self, joints_deg):
        """更新机器人位姿"""
        try:
            # 转换为弧度
            joints_rad = np.deg2rad(joints_deg)
            self.current_joints = np.array(joints_deg)
            self.robot.q = joints_rad

            # 获取所有关节位置
            joint_positions = self.get_joint_positions(joints_rad)

            # 更新连杆
            for i in range(len(joint_positions) - 1):
                p1 = np.array(joint_positions[i])
                p2 = np.array(joint_positions[i + 1])

                direction = p2 - p1
                length = np.linalg.norm(direction)

                if i < len(self.link_meshes):
                    if length > 0.001:
                        center = (p1 + p2) / 2
                        # 创建新的圆柱体几何
                        new_mesh = pv.Cylinder(
                            center=center,
                            direction=direction,
                            radius=0.025,
                            height=length
                        )
                        # 更新现有网格的点
                        self.link_meshes[i].points = new_mesh.points
                        self.link_meshes[i].faces = new_mesh.faces
                    else:
                        # 隐藏零长度连杆
                        self.link_meshes[i].points = np.zeros((1, 3))

            # 隐藏未使用的连杆
            for i in range(len(joint_positions) - 1, len(self.link_meshes)):
                self.link_meshes[i].points = np.zeros((1, 3))

            # 更新关节球体位置
            for i, pos in enumerate(joint_positions[:-1]):
                if i < len(self.joint_meshes):
                    new_sphere = pv.Sphere(center=pos, radius=0.035)
                    self.joint_meshes[i].points = new_sphere.points

            # 隐藏未使用的关节
            for i in range(len(joint_positions) - 1, len(self.joint_meshes)):
                self.joint_meshes[i].points = np.zeros((1, 3))

            # 更新末端执行器位置
            end_pos = joint_positions[-1]
            new_ee = pv.Sphere(center=end_pos, radius=0.03)
            self.end_effector_mesh.points = new_ee.points

            # 更新末端坐标系
            self._update_end_effector_frame(joints_rad)

            # 更新 TCP 文本 (直接修改文本内容，不重建 actor)
            tcp_mm = [p * 1000 for p in end_pos]
            tcp_text = f'TCP: X={tcp_mm[0]:.1f} Y={tcp_mm[1]:.1f} Z={tcp_mm[2]:.1f} mm'
            if self.tcp_text_actor is not None:
                self.tcp_text_actor.SetText(2, tcp_text)  # 2 = upper left position

            # 统一渲染
            self.plotter.render()

        except Exception as e:
            print(f"更新机器人位姿失败: {e}")

    def _update_end_effector_frame(self, joints_rad):
        """更新末端执行器坐标系"""
        T_end = self.robot.fkine(joints_rad)
        pos = T_end.t
        rot = T_end.R

        axis_length = 0.08
        directions = [
            rot @ np.array([1, 0, 0]),  # X
            rot @ np.array([0, 1, 0]),  # Y
            rot @ np.array([0, 0, 1]),  # Z
        ]

        for i, (actor, direction) in enumerate(zip(self.ee_frame_actors, directions)):
            end_point = pos + axis_length * direction
            new_line = pv.Line(pos, end_point)
            # 获取 actor 对应的 mapper 并更新数据
            actor.GetMapper().SetInputData(new_line)

    def get_joint_positions(self, joints_rad):
        """获取所有关节在世界坐标系中的位置"""
        positions = [[0, 0, 0]]  # 基座位置

        # 使用正向运动学计算各关节位置
        for i in range(len(joints_rad)):
            T = self.robot.fkine(joints_rad, end=self.robot.links[i])
            pos = T.t
            positions.append(pos.tolist())

        # 添加末端执行器位置
        T_end = self.robot.fkine(joints_rad)
        positions.append(T_end.t.tolist())

        return positions

    def reset_view(self):
        """重置视角"""
        self.plotter.camera_position = [
            (1.2, -1.2, 0.8),
            (0, 0, 0.3),
            (0, 0, 1)
        ]
        self.plotter.reset_camera()

    def set_test_waypoints(self, waypoints):
        """
        设置测试点位 (从外部更新)

        Args:
            waypoints: [(x, y, z, rz, ry, rx), ...] 单位: mm, 度
        """
        self.test_waypoints = [
            (x, y, z, f"P{i+1}") for i, (x, y, z, rz, ry, rx) in enumerate(waypoints)
        ]
        # 添加点位标记到场景
        self._add_test_waypoints()

    def closeEvent(self, event):
        """关闭事件"""
        self.plotter.close()
        super().closeEvent(event)
