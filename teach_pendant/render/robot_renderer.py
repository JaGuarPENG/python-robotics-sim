"""
机器人 3D 渲染执行器 (支持多模式渲染)
"""

import pyvista as pv
import numpy as np
import os

class RobotRenderer:
    """负责 PyVista 场景中的模型渲染与模式切换"""

    MODE_SIMPLIFIED = 0  # 圆柱体/球体
    MODE_FINE = 1        # STL 模型

    def __init__(self, plotter, model):
        self.plotter = plotter
        self.model = model
        self.mode = self.MODE_SIMPLIFIED
        
        # 简化模型资源
        self.sim_link_meshes = []
        self.sim_joint_meshes = []
        self.sim_ee_mesh = None
        self.sim_actors = []
        
        # 精细模型资源
        self.fine_actors = {} # link_name -> actor
        
        # 公共资源
        self.ee_frame_actors = []
        self.trajectory_actors = []

    def setup_base_scene(self):
        self.plotter.add_axes()
        grid = pv.Plane(i_size=1.2, j_size=1.2)
        self.plotter.add_mesh(grid, color='#2c3e50', opacity=0.5, show_edges=True)

    def create_robot_actors(self):
        """创建所有模式的 Actor (初始隐藏)"""
        self._create_simplified_actors()
        self._create_fine_actors()
        self._create_common_actors()
        self.set_mode(self.MODE_SIMPLIFIED)

    def _create_simplified_actors(self):
        for i in range(7):
            mesh = pv.Cylinder(radius=0.025, height=0.01)
            actor = self.plotter.add_mesh(mesh, color=self.model.link_colors[i%7], smooth_shading=True)
            self.sim_link_meshes.append(mesh)
            self.sim_actors.append(actor)
        
        for i in range(7):
            mesh = pv.Sphere(radius=0.035)
            actor = self.plotter.add_mesh(mesh, color='#ecf0f1')
            self.sim_joint_meshes.append(mesh)
            self.sim_actors.append(actor)

        self.sim_ee_mesh = pv.Sphere(radius=0.03)
        self.sim_actors.append(self.plotter.add_mesh(self.sim_ee_mesh, color='#00FF00'))

    def _create_fine_actors(self):
        """加载 STL 并创建 Actor"""
        if not self.model.urdf_robot: return
        
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        meshes_dir = os.path.join(current_dir, "urdf_export_fine", "urdf_export_fine", "meshes")
        
        for link_name, stl_name in self.model.stl_map.items():
            stl_path = os.path.join(meshes_dir, stl_name)
            if os.path.exists(stl_path):
                mesh = pv.read(stl_path)
                actor = self.plotter.add_mesh(mesh, color='#ecf0f1', smooth_shading=True, specular=0.5)
                actor.SetVisibility(False) # 初始隐藏
                self.fine_actors[link_name] = actor

    def _create_common_actors(self):
        for color in ['#FF0000', '#00FF00', '#0000FF']:
            line = pv.Line((0,0,0), (0.08,0,0))
            self.ee_frame_actors.append(self.plotter.add_mesh(line, color=color, line_width=3))

    def set_mode(self, mode):
        """切换渲染模式"""
        self.mode = mode
        # 切换简化模型可见性
        for actor in self.sim_actors:
            actor.SetVisibility(mode == self.MODE_SIMPLIFIED)
        # 切换精细模型可见性
        for actor in self.fine_actors.values():
            actor.SetVisibility(mode == self.MODE_FINE)
        self.plotter.render()

    def update(self, joints_rad):
        """更新当前模式下的机器人位姿"""
        if self.mode == self.MODE_SIMPLIFIED:
            self._update_simplified(joints_rad)
        else:
            self._update_fine(joints_rad)
        
        # 更新公共末端坐标系
        ee_pos, ee_rot = self.model.get_ee_pose(joints_rad, use_urdf=(self.mode == self.MODE_FINE))
        axes = [ee_rot[:, 0], ee_rot[:, 1], ee_rot[:, 2]]
        for i, (actor, direction) in enumerate(zip(self.ee_frame_actors, axes)):
            new_line = pv.Line(ee_pos, ee_pos + 0.08 * direction)
            actor.GetMapper().SetInputData(new_line)

        self.plotter.render()

    def _update_simplified(self, joints_rad):
        positions = self.model.get_joint_positions(joints_rad)
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i+1]
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if i < len(self.sim_link_meshes):
                if length > 0.001:
                    new_m = pv.Cylinder(center=(p1+p2)/2, direction=vec, radius=0.025, height=length)
                    self.sim_link_meshes[i].points = new_m.points
                else:
                    self.sim_link_meshes[i].points *= 0
        for i, pos in enumerate(positions[:-1]):
            if i < len(self.sim_joint_meshes):
                self.sim_joint_meshes[i].points = pv.Sphere(center=pos, radius=0.035).points
        ee_pos, _ = self.model.get_ee_pose(joints_rad)
        self.sim_ee_mesh.points = pv.Sphere(center=ee_pos, radius=0.03).points

    def _update_fine(self, joints_rad):
        """通过变换矩阵更新 STL Actor"""
        if not self.model.urdf_robot: return
        for link_name, actor in self.fine_actors.items():
            target_link = next((l for l in self.model.urdf_robot.links if l.name == link_name), None)
            if target_link:
                T = self.model.urdf_robot.fkine(joints_rad, end=target_link)
                # 使用 VTK 矩阵更新 Actor 位姿 (最高效)
                actor.user_matrix = T.A

    def set_trajectory(self, points_mm):
        self.clear_trajectory()
        if len(points_mm) < 2: return
        points_m = np.array(points_mm) / 1000.0
        lines = pv.lines_from_points(points_m)
        self.trajectory_actors.append(self.plotter.add_mesh(lines, color='#00FFFF', line_width=4))

    def clear_trajectory(self):
        for a in self.trajectory_actors: self.plotter.remove_actor(a)
        self.trajectory_actors = []