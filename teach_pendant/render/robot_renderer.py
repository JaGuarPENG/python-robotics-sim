"""
机器人 3D 渲染执行器 (仅高精度 STL 模型版)
"""

import pyvista as pv
import numpy as np
import os

class RobotRenderer:
    """负责 PyVista 场景中的高精度模型渲染 (带 50mm 视觉补偿)"""

    def __init__(self, plotter, model):
        self.plotter = plotter
        self.model = model
        self.fine_actors = {}
        self.trajectory_actors = []
        self.actual_path_actors = []
        self.actual_points = []

    def setup_base_scene(self):
        self.plotter.add_axes()
        # 创建一个 50m x 50m 的超大网格，并设置较高的分辨率保持每个网格的间距合理
        grid = pv.Plane(i_size=50, j_size=50, i_resolution=250, j_resolution=250)
        self.plotter.add_mesh(grid, color='#2c3e50', opacity=0.5, show_edges=True)

    def create_robot_actors(self):
        """加载 STL 并创建 Actor (已移除 link6 法兰盘)"""
        if not self.model.urdf_robot: return
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        meshes_dir = os.path.join(current_dir, "urdf_export_fine", "urdf_export_fine", "meshes")
        
        for link_name, stl_name in self.model.stl_map.items():
            stl_path = os.path.join(meshes_dir, stl_name)
            if os.path.exists(stl_path):
                mesh = pv.read(stl_path)
                actor = self.plotter.add_mesh(mesh, color='#ecf0f1', smooth_shading=True, specular=0.5)
                self.fine_actors[link_name] = actor

        # 恢复：创建公共末端轴 (RGB)
        self.ee_frame_actors = []
        for color in ['#FF0000', '#00FF00', '#0000FF']:
            line = pv.Line((0,0,0), (0.08,0,0)) # 80mm 长
            actor = self.plotter.add_mesh(line, color=color, line_width=5)
            self.ee_frame_actors.append(actor)

    def update(self, joints_rad):
        """全量更新 (包含模型与坐标轴，并应用 50mm 下沉)"""
        # 1. 更新 STL 模型
        z_offset_mtx = np.eye(4)
        z_offset_mtx[2, 3] = -0.05
        
        for link_name, actor in self.fine_actors.items():
            target_link = next((l for l in self.model.urdf_robot.links if l.name == link_name), None)
            if target_link:
                T = self.model.urdf_robot.fkine(joints_rad, end=target_link)
                actor.user_matrix = z_offset_mtx @ T.A
        
        # 2. 恢复：更新末端坐标系
        ee_pos, ee_rot = self.model.get_ee_pose(joints_rad, use_urdf=True)
        # 同步下移 50mm
        ee_pos_disp = ee_pos - np.array([0, 0, 0.05])
        
        axes = [ee_rot[:, 0], ee_rot[:, 1], ee_rot[:, 2]]
        for i, (actor, direction) in enumerate(zip(self.ee_frame_actors, axes)):
            new_line = pv.Line(ee_pos_disp, ee_pos_disp + 0.08 * direction)
            actor.GetMapper().SetInputData(new_line)

        self.plotter.render()

    def set_trajectory(self, points_mm):
        self.clear_trajectory()
        points_m = (np.array(points_mm)[:, :3]) / 1000.0
        lines = pv.lines_from_points(points_m)
        self.trajectory_actors.append(self.plotter.add_mesh(lines, color='#00FFFF', line_width=4))

    def add_actual_point(self, pos_m):
        display_pos = np.array([pos_m[0], pos_m[1], pos_m[2] - 0.05])
        self.actual_points.append(display_pos)
        if len(self.actual_points) > 2:
            line = pv.lines_from_points(np.array([self.actual_points[-2], self.actual_points[-1]]))
            self.actual_path_actors.append(self.plotter.add_mesh(line, color='#ff4757', line_width=3))
        if len(self.actual_path_actors) > 500:
            self.plotter.remove_actor(self.actual_path_actors.pop(0))

    def clear_actual_path(self):
        for a in self.actual_path_actors: self.plotter.remove_actor(a)
        self.actual_path_actors = []; self.actual_points = []

    def clear_trajectory(self):
        for a in self.trajectory_actors: self.plotter.remove_actor(a)
        self.trajectory_actors = []
