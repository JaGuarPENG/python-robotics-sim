"""
STEP 数模加载器：STEP -> STL -> Trimesh
用于 PyVista / Open3D 等渲染管线的前置转换
"""
import os
import cadquery as cq
import trimesh
import numpy as np


def step_to_mesh(step_path: str, cache_dir: str = "cache/mesh", target_faces: int = 3000):
    """
    读取 STEP 文件并转换为三角网格。
    会自动缓存 STL 避免重复转换。
    :param step_path:   .step / .stp 文件路径
    :param cache_dir:   STL 缓存目录
    :param target_faces: 网格简化目标面数 (设为极大值可保留原始精度)
    """
    base = os.path.splitext(os.path.basename(step_path))[0]
    stl_path = os.path.join(cache_dir, f"{base}.stl")

    # 1) STEP -> STL (缓存)
    if not os.path.exists(stl_path):
        os.makedirs(cache_dir, exist_ok=True)
        shape = cq.importers.importStep(step_path)
        cq.exporters.export(shape, stl_path)
        print(f"[STEP] 已缓存 STL: {stl_path}")

    # 2) 读取三角网格
    mesh = trimesh.load_mesh(stl_path)

    # 3) 网格简化 (可选)
    n_faces = len(mesh.faces)
    if n_faces > target_faces:
        reduction = 1.0 - (target_faces / n_faces)
        mesh = mesh.simplify_quadric_decimation(reduction)
        print(f"[STEP] 网格已简化至 {len(mesh.faces)} 面")

    return mesh
