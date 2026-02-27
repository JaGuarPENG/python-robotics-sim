
import numpy as np
import roboticstoolbox as rtb
from tools.robot_dh import create_ka_ur

# Create robot
robot = create_ka_ur()
home_joints = [0, -15, 105, 0, -90, 0]
joints_rad = np.deg2rad(home_joints)

# Use forward kinematics to get the pose
T = robot.fkine(joints_rad)
print(f"Home Pos: {T.t}")
print(f"Home RPY (deg): {T.rpy(unit='deg', order='zyx')}")
