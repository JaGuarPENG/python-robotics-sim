"""
轨迹生成器面板 - 两点直线插补生成CSV轨迹
"""

import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class TrajectoryGeneratorPanel(QWidget):
    """两点直线轨迹生成器面板"""
    
    # 信号：轨迹生成完成
    trajectory_generated = pyqtSignal(list)  # 发射轨迹点列表
    
    def __init__(self, robot_view=None, controller=None):
        super().__init__()
        self.robot_view = robot_view
        self.controller = controller
        
        # 存储起点和终点
        self.start_point = None  # [x, y, z, rx, ry, rz]
        self.end_point = None
        self.generated_points = []
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # ==================== 起点设置 ====================
        start_group = QGroupBox("起点 (Start Point)")
        start_layout = QVBoxLayout()
        
        # 起点坐标输入（仅位置）
        start_coord_layout = QHBoxLayout()
        self.start_inputs = []
        labels = ['X(mm):', 'Y(mm):', 'Z(mm):']
        defaults = [300.0, 0.0, 200.0]
        
        for i, (label, default) in enumerate(zip(labels, defaults)):
            start_coord_layout.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setDecimals(2)
            spin.setValue(default)
            spin.setMinimumWidth(120)
            self.start_inputs.append(spin)
            start_coord_layout.addWidget(spin)
        
        # 姿态固定为 180, 0, 0
        start_coord_layout.addWidget(QLabel("  姿态固定:"))
        self.start_orientation_label = QLabel("Rx=180°, Ry=0°, Rz=0°")
        self.start_orientation_label.setStyleSheet("color: #3498db; font-weight: bold;")
        start_coord_layout.addWidget(self.start_orientation_label)
        
        start_coord_layout.addStretch()
        start_layout.addLayout(start_coord_layout)
        
        # 起点操作按钮
        start_btn_layout = QHBoxLayout()
        self.set_start_current_btn = QPushButton("📍 设为当前位置")
        self.set_start_current_btn.clicked.connect(self.set_start_from_current)
        start_btn_layout.addWidget(self.set_start_current_btn)
        
        self.pick_start_btn = QPushButton("🔍 从3D视图选取")
        self.pick_start_btn.clicked.connect(lambda: self.pick_from_view('start'))
        self.pick_start_btn.setEnabled(False)  # 暂时禁用，需要3D视图支持
        start_btn_layout.addWidget(self.pick_start_btn)
        
        start_btn_layout.addStretch()
        start_layout.addLayout(start_btn_layout)
        start_group.setLayout(start_layout)
        layout.addWidget(start_group)
        
        # ==================== 终点设置 ====================
        end_group = QGroupBox("终点 (End Point)")
        end_layout = QVBoxLayout()
        
        # 终点坐标输入（仅位置）
        end_coord_layout = QHBoxLayout()
        self.end_inputs = []
        labels = ['X(mm):', 'Y(mm):', 'Z(mm):']
        defaults = [350.0, 0.0, 200.0]
        
        for i, (label, default) in enumerate(zip(labels, defaults)):
            end_coord_layout.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setDecimals(2)
            spin.setValue(default)
            spin.setMinimumWidth(120)
            self.end_inputs.append(spin)
            end_coord_layout.addWidget(spin)
        
        # 姿态固定为 180, 0, 0
        end_coord_layout.addWidget(QLabel("  姿态固定:"))
        self.end_orientation_label = QLabel("Rx=180°, Ry=0°, Rz=0°")
        self.end_orientation_label.setStyleSheet("color: #3498db; font-weight: bold;")
        end_coord_layout.addWidget(self.end_orientation_label)
        
        end_coord_layout.addStretch()
        end_layout.addLayout(end_coord_layout)
        
        # 终点操作按钮
        end_btn_layout = QHBoxLayout()
        self.set_end_current_btn = QPushButton("📍 设为当前位置")
        self.set_end_current_btn.clicked.connect(self.set_end_from_current)
        end_btn_layout.addWidget(self.set_end_current_btn)
        
        self.pick_end_btn = QPushButton("🔍 从3D视图选取")
        self.pick_end_btn.clicked.connect(lambda: self.pick_from_view('end'))
        self.pick_end_btn.setEnabled(False)
        end_btn_layout.addWidget(self.pick_end_btn)
        
        end_btn_layout.addStretch()
        end_layout.addLayout(end_btn_layout)
        end_group.setLayout(end_layout)
        layout.addWidget(end_group)
        
        # ==================== 插补参数 ====================
        interp_group = QGroupBox("插补参数")
        interp_layout = QHBoxLayout()
        
        # 点数
        interp_layout.addWidget(QLabel("插补点数:"))
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(2, 1000)
        self.num_points_spin.setValue(50)
        self.num_points_spin.setMinimumWidth(80)
        interp_layout.addWidget(self.num_points_spin)
        
        # 时间间隔
        interp_layout.addWidget(QLabel("时间间隔(s):"))
        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setRange(0.001, 1.0)
        self.time_step_spin.setDecimals(3)
        self.time_step_spin.setValue(0.01)
        self.time_step_spin.setMinimumWidth(80)
        interp_layout.addWidget(self.time_step_spin)
        
        # 姿态固定提示
        orientation_info = QLabel("⚠ 姿态固定: Rx=180°, Ry=0°, Rz=0°")
        orientation_info.setStyleSheet("color: #e74c3c; font-weight: bold;")
        interp_layout.addWidget(orientation_info)
        
        interp_layout.addStretch()
        interp_group.setLayout(interp_layout)
        layout.addWidget(interp_group)
        
        # ==================== 操作按钮 ====================
        action_group = QGroupBox("操作")
        action_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("👁 预览轨迹")
        self.preview_btn.clicked.connect(self.preview_trajectory)
        self.preview_btn.setStyleSheet("background-color: #3498db; color: white;")
        action_layout.addWidget(self.preview_btn)
        
        self.generate_btn = QPushButton("⚙ 生成轨迹")
        self.generate_btn.clicked.connect(self.generate_trajectory)
        self.generate_btn.setStyleSheet("background-color: #2ecc71; color: white;")
        action_layout.addWidget(self.generate_btn)
        
        self.clear_btn = QPushButton("🗑 清除")
        self.clear_btn.clicked.connect(self.clear_trajectory)
        action_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("💾 保存CSV")
        self.save_btn.clicked.connect(self.save_csv)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("background-color: #f39c12; color: white;")
        action_layout.addWidget(self.save_btn)
        
        action_layout.addStretch()
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # ==================== 预览表格 ====================
        table_group = QGroupBox("轨迹点预览")
        table_layout = QVBoxLayout()
        
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(7)
        self.points_table.setHorizontalHeaderLabels(
            ['序号', 'X(mm)', 'Y(mm)', 'Z(mm)', 'Rx(°)', 'Ry(°)', 'Rz(°)']
        )
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.points_table.setMaximumHeight(300)
        table_layout.addWidget(self.points_table)
        
        # 统计信息
        self.stats_label = QLabel("轨迹长度: -- mm | 预计时间: -- s")
        table_layout.addWidget(self.stats_label)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        layout.addStretch()
    
    def set_start_from_current(self):
        """从机器人当前位置设置起点（只取位置，姿态固定180,0,0）"""
        if self.controller and self.controller.actual_tcp:
            tcp = self.controller.actual_tcp
            # 只取前3个值（位置），姿态固定为180,0,0
            for i in range(3):
                self.start_inputs[i].setValue(tcp[i])
            # 保存完整点（位置+固定姿态）
            self.start_point = [tcp[0], tcp[1], tcp[2], 180.0, 0.0, 0.0]
        else:
            QMessageBox.warning(self, "警告", "无法获取机器人当前位置，请确保机器人已连接并使能")
    
    def set_end_from_current(self):
        """从机器人当前位置设置终点（只取位置，姿态固定180,0,0）"""
        if self.controller and self.controller.actual_tcp:
            tcp = self.controller.actual_tcp
            # 只取前3个值（位置），姿态固定为180,0,0
            for i in range(3):
                self.end_inputs[i].setValue(tcp[i])
            # 保存完整点（位置+固定姿态）
            self.end_point = [tcp[0], tcp[1], tcp[2], 180.0, 0.0, 0.0]
        else:
            QMessageBox.warning(self, "警告", "无法获取机器人当前位置，请确保机器人已连接并使能")
    
    def pick_from_view(self, point_type):
        """从3D视图中选取点（待实现）"""
        QMessageBox.information(self, "提示", "请在3D视图中点击选择点位\n(功能开发中)")
    
    def get_start_point(self):
        """获取起点坐标（位置+固定姿态180,0,0）"""
        pos = [spin.value() for spin in self.start_inputs]
        return pos + [180.0, 0.0, 0.0]  # 固定姿态
    
    def get_end_point(self):
        """获取终点坐标（位置+固定姿态180,0,0）"""
        pos = [spin.value() for spin in self.end_inputs]
        return pos + [180.0, 0.0, 0.0]  # 固定姿态
    
    def linear_interpolate(self, p1, p2, num_points):
        """
        直线插补（姿态固定为 Rx=180, Ry=0, Rz=0）
        p1, p2: [x, y, z, rx, ry, rz] - 但姿态会被忽略，使用固定值
        返回: 插值点列表
        """
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)  # 0 到 1
            
            # 位置线性插值
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            z = p1[2] + (p2[2] - p1[2]) * t
            
            # 姿态固定为 180, 0, 0
            rx, ry, rz = 180.0, 0.0, 0.0
            
            points.append([x, y, z, rx, ry, rz])
        
        return points
    
    def calculate_distance(self, p1, p2):
        """计算两点之间的距离"""
        return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
    
    def preview_trajectory(self):
        """预览轨迹"""
        p1 = self.get_start_point()
        p2 = self.get_end_point()
        
        num_points = self.num_points_spin.value()
        
        # 生成轨迹点（姿态固定180,0,0）
        self.generated_points = self.linear_interpolate(p1, p2, num_points)
        
        # 更新表格
        self.update_table()
        
        # 更新统计
        distance = self.calculate_distance(p1, p2)
        time_step = self.time_step_spin.value()
        total_time = (num_points - 1) * time_step
        self.stats_label.setText(f"直线距离: {distance:.2f} mm | 点数: {num_points} | 预计时间: {total_time:.2f} s")
        
        # 在3D视图中显示
        if self.robot_view:
            display_points = [(p[0], p[1], p[2]) for p in self.generated_points]
            self.robot_view.set_trajectory(display_points)
        
        self.save_btn.setEnabled(True)
        
        # 发射信号
        self.trajectory_generated.emit(self.generated_points)
    
    def generate_trajectory(self):
        """生成轨迹（带确认）"""
        self.preview_trajectory()
        
        if self.generated_points:
            QMessageBox.information(
                self, 
                "生成成功", 
                f"轨迹已生成！\n共 {len(self.generated_points)} 个点\n"
                f"可以使用【保存CSV】按钮导出文件"
            )
    
    def update_table(self):
        """更新轨迹点表格"""
        self.points_table.setRowCount(len(self.generated_points))
        
        for i, point in enumerate(self.generated_points):
            self.points_table.setItem(i, 0, QTableWidgetItem(str(i)))
            for j, val in enumerate(point):
                self.points_table.setItem(i, j+1, QTableWidgetItem(f"{val:.2f}"))
        
        # 滚动到顶部
        self.points_table.scrollToTop()
    
    def clear_trajectory(self):
        """清除轨迹"""
        self.generated_points = []
        self.points_table.setRowCount(0)
        self.stats_label.setText("轨迹长度: -- mm | 预计时间: -- s")
        self.save_btn.setEnabled(False)
        
        if self.robot_view:
            self.robot_view.clear_trajectory()
    
    def save_csv(self):
        """保存为CSV文件"""
        if not self.generated_points:
            QMessageBox.warning(self, "警告", "没有可保存的轨迹数据")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存轨迹文件",
            os.path.join("csv", "linear_trajectory_generated.csv"),
            "CSV文件 (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            # 创建DataFrame
            df = pd.DataFrame(
                self.generated_points,
                columns=['x', 'y', 'z', 'rx', 'ry', 'rz']
            )
            
            # 添加时间戳列
            time_step = self.time_step_spin.value()
            df['时间戳(秒)'] = [i * time_step for i in range(len(df))]
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            QMessageBox.information(
                self, 
                "保存成功", 
                f"轨迹已保存到:\n{file_path}\n\n共 {len(df)} 个点位"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")
