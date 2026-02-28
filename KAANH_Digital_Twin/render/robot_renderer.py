"""
机器人 3D 渲染执行器 (仅高精度 STL 模型版)
"""

import pyvista as pv
import numpy as np
import os
import time

class RobotRenderer:
    """负责 PyVista 场景中的高精度模型渲染 (带 50mm 视觉补偿)"""

    def __init__(self, plotter, model):
        self.plotter = plotter
        self.model = model
        self.fine_actors = {}
        self.trajectory_actors = []
        self.actual_path_actors = []
        self.actual_points = []
        self.fov_actor = None
        
        # --- Conveyor objects properties ---
        self.belt_objects = []
        self.belt_centers = []
        self.belt_states = [0] * 10  # 0: WAITING (white), 1: TRACKING (green), 2: REACHED (red)
        self.belt_speed = 0.05  # Default 0.05 m/s
        self.last_time = time.time()
        # 初始化 Y 坐标并加入随机扰动，打破绝对等距
        self.obj_y_coords = np.linspace(-2.5, 2.5, 10, endpoint=False) + np.random.uniform(-0.15, 0.15, 10)
        # 初始化随机的 X 坐标 (传送带宽度范围约为 0.45 到 0.75)
        self.obj_x_coords = np.random.uniform(0.45, 0.75, 10)

    def setup_base_scene(self):
        self.plotter.add_axes()
        # 创建一个 50m x 50m 的超大网格，并设置较高的分辨率保持每个网格的间距合理
        grid = pv.Plane(i_size=50, j_size=50, i_resolution=250, j_resolution=250)
        self.plotter.add_mesh(grid, color='#2c3e50', opacity=0.5, show_edges=True)

        # --- Add 5m long conveyor belt (at X=0.6m in front of robot) ---
        # 1. Main body
        conveyor_body = pv.Box(bounds=[0.4, 0.8, -2.5, 2.5, -0.05, 0.15])
        self.plotter.add_mesh(conveyor_body, color='#95a5a6', show_edges=True, edge_color='#2c3e50')

        # 2. Belt surface (dark)
        conveyor_belt = pv.Box(bounds=[0.42, 0.78, -2.5, 2.5, 0.15, 0.16])
        self.plotter.add_mesh(conveyor_belt, color='#2c3e50')

        # 3. Support legs
        for y_pos in [-2.4, 0, 2.4]:
            leg_left = pv.Box(bounds=[0.45, 0.55, y_pos - 0.05, y_pos + 0.05, -0.05, 0.05])
            leg_right = pv.Box(bounds=[0.65, 0.75, y_pos - 0.05, y_pos + 0.05, -0.05, 0.05])
            self.plotter.add_mesh(leg_left, color='#7f8c8d')
            self.plotter.add_mesh(leg_right, color='#7f8c8d')

        # 4. 10 discs on the belt
        for _ in range(10):
            disc = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.1, height=0.05)
            actor = self.plotter.add_mesh(disc, color='#e74c3c', smooth_shading=True)
            self.belt_objects.append(actor)

            # Center marker for each disc
            center_dot = pv.Sphere(radius=0.01)
            center_actor = self.plotter.add_mesh(center_dot, color='#ffffff')
            self.belt_centers.append(center_actor)

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

        # 3. Camera FOV - semi-transparent pyramid (0.7m depth)
        points = np.array([
            [0, 0, 0],             # Vertex (EE)
            [-0.40, -0.40, 0.7],    # Bottom 4 points (0.7 * tan30 ≈ 0.40)
            [0.40, -0.40, 0.7],
            [0.40, 0.40, 0.7],
            [-0.40, 0.40, 0.7],
        ])
        # PyVista faces format: [n_points, i1, i2, ..., n_points, j1, j2, ...]
        faces = [
            3, 0, 1, 2,
            3, 0, 2, 3,
            3, 0, 3, 4,
            3, 0, 4, 1,
            4, 1, 2, 3, 4
        ]
        fov_mesh = pv.PolyData(points, faces)
        self.fov_actor = self.plotter.add_mesh(fov_mesh, color='#3498db', opacity=0.3, show_edges=True, edge_color='#ecf0f1')

        # 4. End-effector Probe (20cm tool)
        # Create a thin cylinder: radius 5mm, length 200mm
        probe_geom = pv.Cylinder(center=(0, 0, 0.1), direction=(0, 0, 1), radius=0.005, height=0.2)
        self.probe_actor = self.plotter.add_mesh(probe_geom, color='#f1c40f', smooth_shading=True)

    def update(self, joints_rad):
        """全量更新 (包含模型、坐标轴与传送带物体，并应用 50mm 下沉)"""
        # Update belt objects positions (looping)
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        dy = self.belt_speed * dt
        for i, actor in enumerate(self.belt_objects):
            self.obj_y_coords[i] += dy
            if self.obj_y_coords[i] > 2.5:
                self.obj_y_coords[i] -= 5.0
                self.obj_x_coords[i] = np.random.uniform(0.45, 0.75)  # 循环时赋予新的随机 X 坐标
                self.belt_states[i] = 0
                self.belt_centers[i].GetProperty().SetColor(1.0, 1.0, 1.0)
            
            T = np.eye(4)
            T[0, 3] = self.obj_x_coords[i]
            T[1, 3] = self.obj_y_coords[i]
            T[2, 3] = 0.185
            actor.user_matrix = T

            T_center = T.copy()
            T_center[2, 3] = 0.211
            self.belt_centers[i].user_matrix = T_center

        # 1. STL Model
        z_offset_mtx = np.eye(4)
        z_offset_mtx[2, 3] = -0.05
        
        for link_name, actor in self.fine_actors.items():
            target_link = next((l for l in self.model.urdf_robot.links if l.name == link_name), None)
            if target_link:
                T = self.model.urdf_robot.fkine(joints_rad, end=target_link)
                actor.user_matrix = z_offset_mtx @ T.A
        
        # 2. End-effector axes
        ee_pos, ee_rot = self.model.get_ee_pose(joints_rad, use_urdf=True)
        ee_pos_disp = ee_pos - np.array([0, 0, 0.05])
        
        # 3. Update FOV actor pose
        T_ee_disp = np.eye(4)
        T_ee_disp[:3, :3] = ee_rot
        T_ee_disp[:3, 3] = ee_pos_disp
        self.fov_actor.user_matrix = T_ee_disp
        
        # 4. Update Probe pose
        if hasattr(self, 'probe_actor'):
            self.probe_actor.user_matrix = T_ee_disp

        # 5. FOV Detection: check if belt centers are inside
        T_inv = np.linalg.inv(T_ee_disp)
        has_tracking = any(s == 1 for s in self.belt_states)

        for i, center_actor in enumerate(self.belt_centers):
            # If already reached, keep it red
            if self.belt_states[i] == 2:
                continue

            world_pos = np.array([self.obj_x_coords[i], self.obj_y_coords[i], 0.211, 1.0])
            local_pos = T_inv @ world_pos
            lx, ly, lz = local_pos[:3]
            
            # Check 60deg FOV pyramid (depth 0.7m, tan(30)≈0.58)
            is_inside = False
            if 0 < lz < 0.7:
                limit = 0.58 * lz
                if abs(lx) < limit and abs(ly) < limit:
                    is_inside = True
            
            if is_inside:
                if self.belt_states[i] == 0 and not has_tracking:
                    self.belt_states[i] = 1
                    has_tracking = True
                    center_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            else:
                if self.belt_states[i] == 1:
                    self.belt_states[i] = 0
                    has_tracking = False
                    center_actor.GetProperty().SetColor(1.0, 1.0, 1.0)

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

    def set_fov_visibility(self, visible):
        """Toggle camera FOV visibility"""
        if self.fov_actor:
            self.fov_actor.SetVisibility(visible)
            self.plotter.render()

    def get_tracking_target(self):
        """Returns the world coordinates [X, Y, Z] and ID of the currently tracked (green) ball, or (None, None)."""
        for i, state in enumerate(self.belt_states):
            if state == 1:
                return np.array([self.obj_x_coords[i], self.obj_y_coords[i], 0.211]), i
        return None, None

    def mark_target_reached(self):
        """Marks the currently tracked ball as reached (red)."""
        for i, state in enumerate(self.belt_states):
            if state == 1:
                self.belt_states[i] = 2
                self.belt_centers[i].GetProperty().SetColor(1.0, 0.0, 0.0)
                return True
        return False
