#!/usr/bin/env python3
"""
Open3D 实时仿真回放器
支持：机器人骨架 + 目标/实际轨迹 + STEP 高精度模型
"""
import numpy as np
import open3d as o3d
import roboticstoolbox as rtb
import time
import os
import sys

# 让 step_loader 可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.step_loader import step_to_mesh


class Open3DReplay:
    def __init__(self, step_path: str = None, step_pose: np.ndarray = None):
        """
        :param step_path: 可选的 STEP/STL 模型路径
        :param step_pose: 4x4 模型位姿变换
        """
        self.robot = rtb.models.UR5()
        self.robot.q = self.robot.qz
        
        # Open3D 窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Simulation Replay (Open3D)", width=1600, height=900)
        
        # 配置渲染
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.line_width = 3.0
        
        # --- 静态场景 ---
        self._add_coord_frame(size=0.2)
        
        # STEP 模型
        self.step_mesh = None
        if step_path:
            self._add_step_model(step_path, step_pose)
        
        # 轨迹线
        self.target_line = o3d.geometry.LineSet()
        self.target_line.paint_uniform_color([1.0, 0.2, 0.2])
        self.vis.add_geometry(self.target_line, False)
        
        self.robot_trail_line = o3d.geometry.LineSet()
        self.robot_trail_line.paint_uniform_color([0.2, 1.0, 0.2])
        self.vis.add_geometry(self.robot_trail_line, False)
        
        # 机器人骨架
        self.robot_skeleton = o3d.geometry.LineSet()
        self.robot_skeleton.paint_uniform_color([0.9, 0.9, 0.2])
        self.vis.add_geometry(self.robot_skeleton, False)
        
        # 目标点
        self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        self.target_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        self.target_sphere.compute_vertex_normals()
        self.vis.add_geometry(self.target_sphere, False)
        
        # 视角：默认看向原点，距离 1.5m
        ctr = self.vis.get_view_control()
        ctr.set_front([0.5, -0.8, -0.4])
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0.3, 0, 0.2])
        ctr.set_zoom(0.5)
        
        self._first = True
    
    def _add_coord_frame(self, size=0.2):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.vis.add_geometry(frame, False)
    
    def _add_step_model(self, path: str, pose: np.ndarray = None):
        print(f"[Open3DReplay] 加载模型: {path}")
        
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.step', '.stp']:
            # 自动转换为 STL 缓存
            tri_mesh = step_to_mesh(path, target_faces=99999999)  # 不简化，高精度
            stl_cache = os.path.join("cache", "mesh", os.path.basename(path).replace(ext, "_o3d.stl"))
            if not os.path.exists(stl_cache):
                tri_mesh.export(stl_cache)
                print(f"[Open3DReplay] 已缓存 STL: {stl_cache}")
            load_path = stl_cache
        else:
            load_path = path
        
        mesh = o3d.io.read_triangle_mesh(load_path)
        if not mesh.has_vertices():
            print("[Open3DReplay] 加载失败")
            return
        mesh.compute_vertex_normals()
        
        if pose is not None:
            mesh = mesh.transform(pose)
        
        # 把模型颜色设为淡灰
        n_verts = len(mesh.vertices)
        colors = np.full((n_verts, 3), [0.7, 0.7, 0.7])
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        self.vis.add_geometry(mesh, False)
        self.step_mesh = mesh
        
        bbox = mesh.get_axis_aligned_bounding_box()
        print(f"[Open3DReplay] STEP 包围盒: {bbox.min_bound} ~ {bbox.max_bound}")
    
    def _update_robot_skeleton(self, q):
        """根据关节角更新机器人骨架"""
        self.robot.q = q
        Ts = self.robot.fkine_all(q)
        positions = [T.t for T in Ts]
        # 过滤掉重复点，保留有效 link 位置
        unique = []
        for p in positions:
            if len(unique) == 0 or np.linalg.norm(p - unique[-1]) > 1e-6:
                unique.append(p)
        
        if len(unique) < 2:
            return
        
        pts = np.array(unique)
        edges = [[i, i+1] for i in range(len(pts)-1)]
        self.robot_skeleton.points = o3d.utility.Vector3dVector(pts)
        self.robot_skeleton.lines = o3d.utility.Vector2iVector(edges)
    
    def run(self, results, realtime=True, speed_factor=1.0):
        """
        运行回放
        :param results: run_simulation 返回的字典
        """
        q_traj = results['q']
        target_pos = np.array(results['target_pos'])
        robot_pos = np.array(results['robot_pos'])
        sim_time = results['time']
        total = len(q_traj)
        
        # 预填充轨迹线
        self.target_line.points = o3d.utility.Vector3dVector(target_pos)
        self.target_line.lines = o3d.utility.Vector2iVector(
            [[i, i+1] for i in range(len(target_pos)-1)]
        )
        self.vis.update_geometry(self.target_line)
        
        print(f"[Open3DReplay] 开始回放: {total} 帧, 时长 {sim_time[-1]:.2f}s")
        
        start_real = time.time()
        start_sim = sim_time[0]
        trail_pts = []
        
        frame_idx = 0
        try:
            while frame_idx < total:
                now = time.time()
                
                if realtime:
                    target_t = start_sim + (now - start_real) * speed_factor
                    while frame_idx < total - 1 and sim_time[frame_idx] < target_t:
                        frame_idx += 1
                    if sim_time[frame_idx] > target_t:
                        wait = (sim_time[frame_idx] - target_t) / speed_factor
                        if wait > 0.002:
                            time.sleep(min(wait, 0.05))
                            continue
                else:
                    frame_idx += max(1, int(total / 400))
                
                if frame_idx >= total:
                    break
                
                # 更新骨架
                self._update_robot_skeleton(q_traj[frame_idx])
                self.vis.update_geometry(self.robot_skeleton)
                
                # 更新目标球位置
                tp = target_pos[frame_idx]
                self.target_sphere.translate(tp - self.target_sphere.get_center(), relative=False)
                self.vis.update_geometry(self.target_sphere)
                
                # 更新实际轨迹
                rp = robot_pos[frame_idx]
                trail_pts.append(rp)
                if len(trail_pts) > 1:
                    self.robot_trail_line.points = o3d.utility.Vector3dVector(np.array(trail_pts))
                    self.robot_trail_line.lines = o3d.utility.Vector2iVector(
                        [[i, i+1] for i in range(len(trail_pts)-1)]
                    )
                    self.vis.update_geometry(self.robot_trail_line)
                
                self.vis.poll_events()
                self.vis.update_renderer()
        
        except KeyboardInterrupt:
            print("\n[Open3DReplay] 用户中断")
        
        print("[Open3DReplay] 回放结束，关闭窗口...")
        self.vis.destroy_window()


if __name__ == "__main__":
    # 快速测试：加载 mochuang.stl 并空跑
    replay = Open3DReplay(step_path=r"cache\mesh\mochuang.stl")
    # 可以在这里调用 replay.run(results) 做真实回放
    print("按 Ctrl+C 或关闭窗口退出")
    while replay.vis.poll_events():
        replay.vis.update_renderer()
        time.sleep(0.05)
    replay.vis.destroy_window()
