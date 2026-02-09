import sys
import os
import numpy as np
import roboticstoolbox as rtb

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_ik.ik_solver import FastIKSolver
from tools.robot_dh import create_ka_ur

def verify_accuracy():
    print("--- IK 精度验证 (PyBullet vs Roboticstoolbox) ---")
    
    # 1. 初始化两个模型
    # PyBullet (FastIK)
    solver = FastIKSolver()
    
    # RTB (Standard DH)
    robot_rtb = create_ka_ur()
    
    # 2. 定义测试点 (主窗口里那个红点)
    # 目标: x=488, y=164, z=957, rx=225, ry=-20, rz=0 (mm, deg)
    target_pos = [0.488, 0.164, 0.957] # m
    target_rpy = [225, -20, 0] # deg
    
    print(f"\n目标位置 (m): {target_pos}")
    print(f"目标姿态 (deg): {target_rpy}")
    
    # 3. 使用 FastIK 解算
    print("\n[1] FastIK 解算中...")
    joints_rad, dt = solver.solve_ik(target_pos, target_rpy)
    
    if joints_rad is None:
        print("FastIK 解算失败!")
        return

    print(f"解算耗时: {dt:.4f} ms")
    print(f"解算关节角 (deg): {[f'{np.rad2deg(j):.2f}' for j in joints_rad]}")
    
    # 4. 使用 RTB 进行正解验证 (FK)
    print("\n[2] RTB 正解验证 (检查模型一致性)...")
    # FK 输入需要是弧度
    T_check = robot_rtb.fkine(joints_rad)
    
    pos_check = T_check.t
    rpy_check = T_check.rpy(unit='deg', order='zyx') # ZYX顺序
    
    print(f"RTB 正解位置 (m): {pos_check}")
    
    # 5. 计算误差
    diff_pos = np.linalg.norm(pos_check - target_pos) * 1000 # mm
    print(f"\n[结论] 模型位置误差: {diff_pos:.4f} mm")
    
    if diff_pos < 1.0:
        print("-> 模型一致性良好 (误差 < 1mm)")
    else:
        print("-> 警告: 模型存在显著差异! 请检查 URDF 生成参数。")
        print(f"   目标 X: {target_pos[0]:.4f} vs 实际 X: {pos_check[0]:.4f}")
        print(f"   目标 Y: {target_pos[1]:.4f} vs 实际 Y: {pos_check[1]:.4f}")
        print(f"   目标 Z: {target_pos[2]:.4f} vs 实际 Z: {pos_check[2]:.4f}")

    solver.close()

if __name__ == "__main__":
    verify_accuracy()
