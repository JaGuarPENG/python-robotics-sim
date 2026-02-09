
import pybullet as p
import numpy as np
import os
import time

class FastIKSolver:
    """
    基于 PyBullet (C++ 后端) 的高性能逆运动学求解器
    """
    def __init__(self, urdf_path=None):
        # 1. 确定 URDF 路径
        if urdf_path is None:
            # 默认使用同目录下的 ka_ur.urdf (由 generate_urdf.py 生成)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(current_dir, "ka_ur.urdf")
        
        self.urdf_path = urdf_path
        
        # 2. 连接 PyBullet (使用 DIRECT 模式，不显示 GUI，纯计算)
        self.client_id = p.connect(p.DIRECT)
        
        # 3. 设置搜索路径以便加载 STL 
        # 因为 URDF 里的路径是 package://...，我们需要让 pybullet 知道在哪里找
        # 这里的技巧是：pybullet 加载时如果找不到 package，会尝试在同一目录下找
        p.setAdditionalSearchPath(os.path.dirname(urdf_path), physicsClientId=self.client_id)
        
        # 4. 加载机器人
        # 我们禁用了碰撞检测和重力，因为只做运动学计算
        self.robot_id = p.loadURDF(urdf_path, 
                                   useFixedBase=True, 
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
                                   physicsClientId=self.client_id)
        
        # 5. 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        self.revolute_joint_indices = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if info[2] == p.JOINT_REVOLUTE:
                self.revolute_joint_indices.append(i)
        
        # 机器人末端执行器索引 (通常是最后一个关节的 Link)
        self.ee_link_index = self.revolute_joint_indices[-1]
        
        print(f"[FastIKSolver] 成功加载机器人，有效关节数: {len(self.revolute_joint_indices)}")

    def solve_ik(self, pos, rpy_deg=None, quat=None, current_joints=None):
        """
        解算逆运动学
        
        Args:
            pos: [x, y, z] 目标位置 (单位: 米)
            rpy_deg: [rx, ry, rz] 目标欧拉角 (单位: 度)
            quat: [x, y, z, w] 目标四元数 (如果提供，则忽略 rpy_deg)
            current_joints: 当前关节角 (弧度)，作为初值种子
            
        Returns:
            joint_angles: 弧度制的 6 个关节角，如果失败返回 None
        """
        # 1. 姿态处理
        if quat is None and rpy_deg is not None:
            # 将度数转为弧度，并转为四元数
            rpy_rad = [np.deg2rad(a) for i, a in enumerate(rpy_deg)]
            # 注意：PyBullet 使用的是 [x, y, z, w] 顺序的四元数
            # 这里的 RPY 顺序需要与您的机器人坐标系定义匹配
            quat = p.getQuaternionFromEuler(rpy_rad)
        
        # 2. 设置初值 (有助于找到连续解)
        if current_joints is not None:
            for i, idx in enumerate(self.revolute_joint_indices):
                p.resetJointState(self.robot_id, idx, current_joints[i], physicsClientId=self.client_id)
        
        # 3. 调用 C++ 后端求解
        # PyBullet 的求解器非常快速
        t_start = time.perf_counter()
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            targetPosition=pos,
            targetOrientation=quat,
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client_id
        )
        t_end = time.perf_counter()
        
        # 4. 提取结果 (只返回转动关节的角度)
        result = [joint_angles[i] for i in range(len(self.revolute_joint_indices))]
        
        # 耗时统计 (ms)
        elapsed_ms = (t_end - t_start) * 1000
        
        return result, elapsed_ms

    def close(self):
        p.disconnect(self.client_id)

# 测试代码
if __name__ == "__main__":
    solver = FastIKSolver()
    
    # 测试点: [0.4, 0, 0.4]
    target_pos = [0.4, 0.1, 0.5]
    target_rpy = [0, 0, 0] # 度
    
    print(f"\n开始测试解算点: {target_pos}")
    joints, dt = solver.solve_ik(target_pos, target_rpy)
    
    if joints:
        print(f"解算成功!")
        print(f"关节角 (度): {[f'{np.rad2deg(j):.2f}' for j in joints]}")
        print(f"解算耗时: {dt:.4f} ms")
    else:
        print("解算失败")
        
    solver.close()
