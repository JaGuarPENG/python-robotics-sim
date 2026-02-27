
import numpy as np
from fast_ik.ik_solver import FastIKSolver

# Initialize solver
solver = FastIKSolver()
home_joints = [0, -15, 105, 0, -90, 0]
joints_rad = np.deg2rad(home_joints)

# Use forward kinematics to get the pose
# Note: In the project's FastIKSolver or robot_dh, there's likely an fkine
from tools.robot_dh import RobotDH
robot = RobotDH()
T = robot.fkine(joints_rad)
print(f"Home Pos: {T.t}")
print(f"Home RPY (deg): {T.rpy(unit='deg', order='zyx')}")
