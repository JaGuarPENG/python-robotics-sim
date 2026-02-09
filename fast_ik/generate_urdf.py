
import os
import math

def generate_urdf(filepath="ka_ur.urdf"):
    """
    根据 robot_dh.py 的逻辑生成 URDF 文件
    注意：所有长度单位为米，角度为弧度
    """
    
    # 辅助函数：生成 Link
    def link_xml(name, radius=0.05, length=0.1):
        return f"""
  <link name="{name}">
    <visual>
      <geometry>
        <cylinder length="{length}" radius="{radius}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
  </link>
"""

    # 辅助函数：生成 Joint
    def joint_xml(name, type, parent, child, xyz, rpy, axis="0 0 1", limits=(-3.14, 3.14)):
        return f"""
  <joint name="{name}" type="{type}">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <axis xyz="{axis}"/>
    <limit lower="{limits[0]}" upper="{limits[1]}" effort="100" velocity="3.0"/>
  </joint>
"""

    # 参数定义 (mm -> m)
    L1_Z = 0.1693
    
    L2_Y = 0.1793
    L2_Rx = -math.pi / 2
    
    L3_Y = -0.625
    
    # J3 -> J4: tz(-0.1626) * ty(-0.595)
    L4_Z = -0.1626
    L4_Y = -0.595
    
    # J4 -> J5: tz(0.1052) * Rx(90)
    L5_Z = 0.1052
    L5_Rx = math.pi / 2
    
    # J5 -> J6: tz(0.1052) * ty(0.110) * Rx(-90)
    # 注意：在 robot_dh.py 中是 l5_dummy(tz) -> l6(ty * Rx)
    # 这里的复合变换: tz(0.1052) * ty(0.110) * Rx(-90)
    # 在 Joint 5 frame 下:
    # Z轴朝上. tz(0.1052) 沿Z轴动. ty(0.110) 沿Y轴动. 
    # 所以 xyz = "0 0.110 0.1052"
    L6_Z_Offset = 0.1052
    L6_Y = 0.110
    L6_Rx = -math.pi / 2
    
    urdf_content = f"""<?xml version="1.0"?>
<robot name="ka_ur">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.05"/>
      </geometry>
    </visual>
  </link>

  {link_xml("link1", 0.06, 0.15)}
  {link_xml("link2", 0.05, 0.6)}
  {link_xml("link3", 0.05, 0.5)}
  {link_xml("link4", 0.04, 0.1)}
  {link_xml("link5", 0.04, 0.1)}
  {link_xml("link6", 0.03, 0.05)}
  
  <!-- TCP Link (End Effector) -->
  <link name="tool0"/>

  <!-- Joint 1: Base -> Link1 -->
  <!-- Logic: tz({L1_Z}) * Rz -->
  {joint_xml("joint1", "revolute", "base_link", "link1", 
             f"0 0 {L1_Z}", "0 0 0")}

  <!-- Joint 2: Link1 -> Link2 -->
  <!-- Logic: ty({L2_Y}) * Rx(-90) * Rz -->
  {joint_xml("joint2", "revolute", "link1", "link2", 
             f"0 {L2_Y} 0", f"{L2_Rx} 0 0")}

  <!-- Joint 3: Link2 -> Link3 -->
  <!-- Logic: ty({L3_Y}) * Rz -->
  {joint_xml("joint3", "revolute", "link2", "link3", 
             f"0 {L3_Y} 0", "0 0 0")}

  <!-- Joint 4: Link3 -> Link4 -->
  <!-- Logic: tz({L4_Z}) * ty({L4_Y}) * Rz -->
  {joint_xml("joint4", "revolute", "link3", "link4", 
             f"0 {L4_Y} {L4_Z}", "0 0 0")}

  <!-- Joint 5: Link4 -> Link5 -->
  <!-- Logic: tz({L5_Z}) * Rx(90) * Rz -->
  {joint_xml("joint5", "revolute", "link4", "link5", 
             f"0 0 {L5_Z}", f"{L5_Rx} 0 0")}

  <!-- Joint 6: Link5 -> Link6 -->
  <!-- Logic: tz({L6_Z_Offset}) * ty({L6_Y}) * Rx(-90) * Rz -->
  {joint_xml("joint6", "revolute", "link5", "link6", 
             f"0 {L6_Y} {L6_Z_Offset}", f"{L6_Rx} 0 0")}

  <!-- TCP Joint (Fixed) -->
  <joint name="joint_tcp" type="fixed">
    <parent link="link6"/>
    <child link="tool0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

</robot>
"""
    
    # 写入文件
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(urdf_content)
    print(f"[URDF Generator] 成功生成: {output_path}")

if __name__ == "__main__":
    generate_urdf("ka_ur.urdf")
