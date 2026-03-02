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
    trajectory_generated = pyqtSignal(list, float)  # 发射(轨迹点列表, 时间间隔)
    
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
        
        # 固定频率设置
        interp_layout.addWidget(QLabel("发送频率:"))
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(10, 125)  # 工控机最大125Hz
        self.frequency_spin.setDecimals(0)
        self.frequency_spin.setValue(125)  # 默认最大频率
        self.frequency_spin.setSuffix(" Hz")
        self.frequency_spin.setMinimumWidth(100)
        self.frequency_spin.setToolTip("工控机最大接收频率为125Hz")
        interp_layout.addWidget(self.frequency_spin)
        
        # 计算出的时间间隔（只读显示）
        interp_layout.addWidget(QLabel("时间间隔:"))
        self.time_step_label = QLabel("8.0 ms")
        self.time_step_label.setStyleSheet("color: #3498db; font-weight: bold; min-width: 80px;")
        interp_layout.addWidget(self.time_step_label)
        
        # 速度设置（主要输入）
        interp_layout.addWidget(QLabel("目标速度:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 500.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setValue(50.0)  # 默认50mm/s
        self.speed_spin.setSuffix(" mm/s")
        self.speed_spin.setMinimumWidth(120)
        interp_layout.addWidget(self.speed_spin)
        
        # 速度单位切换
        self.speed_unit_combo = QComboBox()
        self.speed_unit_combo.addItems(["mm/s", "m/s"])
        self.speed_unit_combo.setMinimumWidth(80)
        self.speed_unit_combo.currentIndexChanged.connect(self.on_speed_unit_changed)
        interp_layout.addWidget(self.speed_unit_combo)
        
        # 计算点数按钮
        self.calc_btn = QPushButton("🔄 计算插补点数")
        self.calc_btn.setToolTip("根据速度、距离、固定频率自动计算所需插补点数")
        self.calc_btn.clicked.connect(self.calculate_points_from_speed)
        self.calc_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        interp_layout.addWidget(self.calc_btn)
        
        interp_layout.addStretch()
        interp_group.setLayout(interp_layout)
        layout.addWidget(interp_group)
        
        # 第二行：插补点数显示和姿态提示
        info_layout = QHBoxLayout()
        
        # 插补点数（计算结果，可手动微调）
        info_layout.addWidget(QLabel("插补点数:"))
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(2, 10000)
        self.num_points_spin.setValue(50)
        self.num_points_spin.setMinimumWidth(100)
        self.num_points_spin.valueChanged.connect(self.on_points_changed)
        info_layout.addWidget(self.num_points_spin)
        
        # 预计总时间
        info_layout.addWidget(QLabel("预计总时间:"))
        self.total_time_label = QLabel("-- s")
        self.total_time_label.setStyleSheet("color: #2ecc71; font-weight: bold; min-width: 80px;")
        info_layout.addWidget(self.total_time_label)
        
        # 姿态固定提示
        orientation_info = QLabel("⚠ 姿态固定: Rx=180°, Ry=0°, Rz=0°")
        orientation_info.setStyleSheet("color: #e74c3c; font-weight: bold;")
        info_layout.addWidget(orientation_info)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
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
    
    def get_speed_mm_s(self):
        """获取速度，统一转换为 mm/s"""
        speed = self.speed_spin.value()
        if self.speed_unit_combo.currentText() == "m/s":
            speed = speed * 1000.0  # m/s -> mm/s
        return speed
    
    def set_speed_mm_s(self, speed_mm_s):
        """设置速度（内部统一用mm/s）"""
        if self.speed_unit_combo.currentText() == "m/s":
            self.speed_spin.setValue(speed_mm_s / 1000.0)
        else:
            self.speed_spin.setValue(speed_mm_s)
    
    def on_speed_changed(self):
        """速度改变时，可以选择自动更新时间间隔"""
        pass  # 不自动计算，需要点击按钮才计算
    
    def on_speed_unit_changed(self, index):
        """速度单位改变时，转换数值"""
        if index == 0:  # 切换到 mm/s
            # 当前是 m/s，转换为 mm/s
            current = self.speed_spin.value()
            self.speed_spin.setRange(0.1, 1000.0)
            self.speed_spin.setDecimals(1)
            self.speed_spin.setValue(current * 1000.0)
            self.speed_spin.setSuffix(" mm/s")
        else:  # 切换到 m/s
            # 当前是 mm/s，转换为 m/s
            current = self.speed_spin.value()
            self.speed_spin.setRange(0.0001, 1.0)
            self.speed_spin.setDecimals(4)
            self.speed_spin.setValue(current / 1000.0)
            self.speed_spin.setSuffix(" m/s")
    
    def on_points_changed(self):
        """插补点数改变时，更新预计总时间"""
        frequency = self.frequency_spin.value()
        num_points = self.num_points_spin.value()
        time_step = 1.0 / frequency
        
        # 计算总时间
        total_time = (num_points - 1) * time_step
        self.total_time_label.setText(f"{total_time:.3f} s")
        
        # 更新距离和速度信息
        p1 = self.get_start_point()
        p2 = self.get_end_point()
        distance = self.calculate_distance(p1, p2)
        
        if distance > 0.01:
            actual_speed = distance / total_time
            # 可以在这里显示实际速度
    
    def on_speed_changed(self):
        """速度改变时触发（预留）"""
        pass
    
    def calculate_points_from_speed(self):
        """
        根据设置的速度和固定频率，自动计算所需插补点数
        
        核心公式: 
        - 总时间 = 轨迹长度 / 速度
        - 需要的点数 = 总时间 * 频率 + 1
        
        这样能以固定频率(如125Hz)发送，达到目标速度
        """
        p1 = self.get_start_point()
        p2 = self.get_end_point()
        
        # 计算轨迹长度
        distance = self.calculate_distance(p1, p2)  # mm
        
        if distance < 0.01:  # 距离太小
            QMessageBox.warning(self, "警告", "起点和终点距离太小，无法计算")
            return
        
        # 获取速度 (mm/s)
        speed = self.get_speed_mm_s()
        
        if speed < 0.01:
            QMessageBox.warning(self, "警告", "速度设置太小")
            return
        
        # 获取固定频率
        frequency = self.frequency_spin.value()  # Hz
        time_step = 1.0 / frequency  # 秒
        
        # ============================================
        # 核心计算逻辑
        # ============================================
        # 1. 计算总时间 (由距离和速度决定)
        total_time = distance / speed  # 秒
        
        # 2. 计算需要的插补点数
        # N个点有N-1个间隔，每个间隔1/frequency秒
        # 总时间 = (N-1) * (1/frequency)
        # N = 总时间 * frequency + 1
        num_points = int(total_time * frequency) + 1
        
        # 限制点数范围
        if num_points < 2:
            num_points = 2
        if num_points > 10000:
            # 如果点数太多，提示用户降低频率或提高速度
            QMessageBox.warning(
                self,
                "插补点数过多",
                f"计算需要 {num_points} 个点，超过最大限制(10000)。\n"
                f"建议:\n"
                f"1. 降低发送频率 (当前{frequency}Hz)\n"
                f"2. 提高运动速度 (当前{speed:.1f}mm/s)\n"
                f"3. 缩短运动距离"
            )
            num_points = 10000
        
        # 更新UI
        self.num_points_spin.blockSignals(True)
        self.num_points_spin.setValue(num_points)
        self.num_points_spin.blockSignals(False)
        
        # 更新时间间隔显示
        self.time_step_label.setText(f"{time_step*1000:.1f} ms")
        
        # 计算实际能达到的速度（可能因为取整有微小差异）
        actual_total_time = (num_points - 1) * time_step
        actual_speed = distance / actual_total_time
        
        # 显示信息
        speed_display = speed
        actual_speed_display = actual_speed
        speed_unit = "mm/s"
        if self.speed_unit_combo.currentText() == "m/s":
            speed_display = speed / 1000.0
            actual_speed_display = actual_speed / 1000.0
            speed_unit = "m/s"
        
        self.total_time_label.setText(f"{actual_total_time:.3f} s")
        
        QMessageBox.information(
            self,
            "计算完成",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"轨迹长度: {distance:.2f} mm ({distance/1000:.3f} m)\n"
            f"目标速度: {speed_display:.2f} {speed_unit}\n"
            f"发送频率: {frequency} Hz (固定)\n"
            f"时间间隔: {time_step*1000:.1f} ms\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"计算结果:\n"
            f"  插补点数: {num_points} 点\n"
            f"  预计总时间: {actual_total_time:.3f} s\n"
            f"  实际速度: {actual_speed_display:.2f} {speed_unit}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✓ 以 {frequency}Hz 频率发送 {num_points} 个点\n"
            f"✓ 机器人将以约 {actual_speed_display:.2f} {speed_unit} 运动"
        )
        
        # 触发一次预览更新
        self.preview_trajectory()
    
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
        frequency = self.frequency_spin.value()
        time_step = 1.0 / frequency
        total_time = (num_points - 1) * time_step
        actual_speed = distance / total_time if total_time > 0 else 0
        
        self.stats_label.setText(f"距离: {distance:.2f}mm | 点数: {num_points} | "
                                f"频率: {frequency}Hz | 时间: {total_time:.2f}s | "
                                f"速度: {actual_speed:.1f}mm/s")
        
        # 在3D视图中显示
        if self.robot_view:
            display_points = [(p[0], p[1], p[2]) for p in self.generated_points]
            self.robot_view.set_trajectory(display_points)
        
        self.save_btn.setEnabled(True)
        
        # 发射信号（带上时间间隔）
        self.trajectory_generated.emit(self.generated_points, time_step)
    
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
            
            # 添加时间戳列（根据固定频率计算）
            frequency = self.frequency_spin.value()
            time_step = 1.0 / frequency
            df['时间戳(秒)'] = [i * time_step for i in range(len(df))]
            
            # 添加频率信息到注释
            distance = self.calculate_distance(self.generated_points[0], self.generated_points[-1])
            total_time = (len(self.generated_points) - 1) * time_step
            actual_speed = distance / total_time
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            QMessageBox.information(
                self, 
                "保存成功", 
                f"轨迹已保存到:\n{file_path}\n\n"
                f"共 {len(df)} 个点位\n"
                f"发送频率: {frequency}Hz\n"
                f"时间间隔: {time_step*1000:.1f}ms\n"
                f"预计总时间: {total_time:.2f}s\n"
                f"实际速度: {actual_speed:.1f}mm/s"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")
