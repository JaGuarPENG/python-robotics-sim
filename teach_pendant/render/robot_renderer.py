"""
机器人 3D 渲染执行器 (基于 PyVista)
"""

import pyvista as pv
import numpy as np

class RobotRenderer:
    """负责 PyVista 场景中的网格创建与更新"""

    def __init__(self, plotter, model):
        self.plotter = plotter
        self.model = model
        
        self.link_actors = []
        self.link_meshes = []
        self.joint_meshes = []
        self.ee_actor = None
        self.ee_mesh = None
        self.ee_frame_actors = []
        self.trajectory_actors = []

    def setup_base_scene(self):
        """初始化基础环境"""
        # 坐标轴
        self.plotter.add_axes()
        # 地面网格
        grid = pv.Plane(i_size=1.2, j_size=1.2)
        self.plotter.add_mesh(grid, color='#2c3e50', opacity=0.5, show_edges=True)

    def create_robot_actors(self):
        """预创建机器人所有部件的 Actor"""
        # 连杆
        for i in range(7):
            mesh = pv.Cylinder(radius=0.025, height=0.01)
            actor = self.plotter.add_mesh(mesh, color=self.model.link_colors[i%7], smooth_shading=True)
            self.link_meshes.append(mesh)
            self.link_actors.append(actor)
        
        # 关节
        for i in range(7):
            mesh = pv.Sphere(radius=0.035)
            self.plotter.add_mesh(mesh, color='#ecf0f1')
            self.joint_meshes.append(mesh)

        # 末端执行器
        self.ee_mesh = pv.Sphere(radius=0.03)
        self.ee_actor = self.plotter.add_mesh(self.ee_mesh, color='#00FF00')

        # 末端坐标系
        for color in ['#FF0000', '#00FF00', '#0000FF']:
            line = pv.Line((0,0,0), (0.08,0,0))
            self.ee_frame_actors.append(self.plotter.add_mesh(line, color=color, line_width=3))

    def update(self, joints_rad):
        """全量更新机器人位姿"""
        positions = self.model.get_joint_positions(joints_rad)
        
        # 更新连杆
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i+1]
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if i < len(self.link_meshes):
                if length > 0.001:
                    new_m = pv.Cylinder(center=(p1+p2)/2, direction=vec, radius=0.025, height=length)
                    self.link_meshes[i].points = new_m.points
                else:
                    self.link_meshes[i].points *= 0

        # 更新关节
        for i, pos in enumerate(positions[:-1]):
            if i < len(self.joint_meshes):
                self.joint_meshes[i].points = pv.Sphere(center=pos, radius=0.035).points

        # 更新末端
        ee_pos, ee_rot = self.model.get_ee_pose(joints_rad)
        self.ee_mesh.points = pv.Sphere(center=ee_pos, radius=0.03).points
        
        # 更新坐标系
        axes = [ee_rot[:, 0], ee_rot[:, 1], ee_rot[:, 2]]
        for i, (actor, direction) in enumerate(zip(self.ee_frame_actors, axes)):
            new_line = pv.Line(ee_pos, ee_pos + 0.08 * direction)
            actor.GetMapper().SetInputData(new_line)

        self.plotter.render()

    def set_trajectory(self, points_mm):
        self.clear_trajectory()
        if len(points_mm) < 2: return
        
        points_m = np.array(points_mm) / 1000.0
        lines = pv.lines_from_points(points_m)
        actor = self.plotter.add_mesh(lines, color='#00FFFF', line_width=4)
        self.trajectory_actors.append(actor)

    def clear_trajectory(self):
        for a in self.trajectory_actors:
            self.plotter.remove_actor(a)
        self.trajectory_actors = []
