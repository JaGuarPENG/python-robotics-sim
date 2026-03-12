"""
视觉功能面板 (CSV 文件版)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QMessageBox, QFileDialog,
    QHBoxLayout, QSlider, QLineEdit, QGridLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
import pandas as pd
import os

class VisionPanel(QWidget):
    """视觉功能面板：加载 CSV 轨迹并执行 + 传送带动态追踪"""
    
    trajectory_generated = pyqtSignal(list)
    execution_requested = pyqtSignal()
    udp_execution_requested = pyqtSignal()
    actual_export_requested = pyqtSignal()
    
    # 新增传送带追踪相关信号
    conveyor_tracking_toggled = pyqtSignal(bool)
    conveyor_sim_tracking_toggled = pyqtSignal(bool)
    conveyor_offset_tracking_toggled = pyqtSignal(bool)
    conveyor_speed_changed = pyqtSignal(float)
    conveyor_hover_only_toggled = pyqtSignal(bool)
    conveyor_udp_follower_tracking_toggled = pyqtSignal(bool)
    
    # 单点遥操作移动信号: (target_x, target_y, target_z, target_rx, target_ry, target_rz)
    # 传入的是目标绝对坐标，由主窗口计算偏移量
    single_point_move_requested = pyqtSignal(float, float, float, float, float, float)
    
    # 获取当前坐标信号
    get_current_position_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_path = 'csv/vision_trajectory.csv'
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- 第一部分：CSV 轨迹导入与执行 ---
        csv_group = QGroupBox("视觉引导 (CSV 导入模式)")
        csv_layout = QVBoxLayout(csv_group)

        self.info_label = QLabel("1. 运行 verify_vision.py 并保存 CSV\n2. 点击下方按钮导入并执行")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #bdc3c7; font-size: 11px;")
        csv_layout.addWidget(self.info_label)

        # 按钮区
        self.load_btn = QPushButton("第一步：加载视觉 CSV 轨迹")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.load_btn.clicked.connect(self.on_load_clicked)
        csv_layout.addWidget(self.load_btn)

        self.execute_btn = QPushButton("第二步：执行运动 (IK 模式)")
        self.execute_btn.setMinimumHeight(40)
        self.execute_btn.setEnabled(False)
        self.execute_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        self.execute_btn.clicked.connect(self.execution_requested.emit)
        csv_layout.addWidget(self.execute_btn)

        self.execute_udp_btn = QPushButton("第二步：执行运动 (UDP 增量模式)")
        self.execute_udp_btn.setMinimumHeight(40)
        self.execute_udp_btn.setEnabled(False)
        self.execute_udp_btn.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
        self.execute_udp_btn.clicked.connect(self.on_udp_execution_clicked)
        csv_layout.addWidget(self.execute_udp_btn)

        self.export_btn = QPushButton("第三步：导出实际运行 CSV")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.export_btn.clicked.connect(self.actual_export_requested.emit)
        csv_layout.addWidget(self.export_btn)
        
        # 结果显示
        self.result_label = QLabel("等待导入...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-family: Courier New; color: #f1c40f; background: #2c3e50; padding: 5px; border-radius: 3px;")
        csv_layout.addWidget(self.result_label)

        layout.addWidget(csv_group)

        # --- 第二部分：传送带动态追踪控制 ---
        tracking_group = QGroupBox("传送带动态追踪")
        tracking_layout = QVBoxLayout(tracking_group)

        self.tracking_btn = QPushButton("开启传送带追踪 (绝对位置PID)")
        self.tracking_btn.setCheckable(True)
        self.tracking_btn.setMinimumHeight(50)
        self.tracking_btn.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.tracking_btn.toggled.connect(self.on_tracking_toggled)
        tracking_layout.addWidget(self.tracking_btn)

        # 新增仿真算法追踪按钮
        self.sim_tracking_btn = QPushButton("开启传送带追踪 (仿真速度PI控制)")
        self.sim_tracking_btn.setCheckable(True)
        self.sim_tracking_btn.setMinimumHeight(50)
        self.sim_tracking_btn.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #2980b9; }
        """)
        self.sim_tracking_btn.toggled.connect(self.on_sim_tracking_toggled)
        tracking_layout.addWidget(self.sim_tracking_btn)

        # 新增自适应隐性 Offset 追踪按钮
        self.offset_tracking_btn = QPushButton("开启传送带追踪 (自适应隐性Offset)")
        self.offset_tracking_btn.setCheckable(True)
        self.offset_tracking_btn.setMinimumHeight(50)
        self.offset_tracking_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #8e44ad; }
        """)
        self.offset_tracking_btn.toggled.connect(self.on_offset_tracking_toggled)
        tracking_layout.addWidget(self.offset_tracking_btn)

        # 速度调节
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("传送带速度:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 200) # 0.00 to 0.20 m/s
        self.speed_slider.setValue(50)     # 默认 0.05 m/s
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_val_label = QLabel("0.05 m/s")
        self.speed_val_label.setFixedWidth(100)
        speed_layout.addWidget(self.speed_val_label)
        tracking_layout.addLayout(speed_layout)

        # 新增：仅悬停模式按钮
        self.hover_only_btn = QPushButton("开启仅悬停追踪 (不触碰)")
        self.hover_only_btn.setCheckable(True)
        self.hover_only_btn.setMinimumHeight(50)
        self.hover_only_btn.setStyleSheet("""
            QPushButton { background-color: #34495e; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #2c3e50; }
        """)
        self.hover_only_btn.toggled.connect(self.on_hover_only_toggled)
        tracking_layout.addWidget(self.hover_only_btn)

        # 新增：UDP follower_cart 追踪按钮
        self.udp_follower_tracking_btn = QPushButton("UDP追踪小球")
        self.udp_follower_tracking_btn.setCheckable(True)
        self.udp_follower_tracking_btn.setMinimumHeight(50)
        self.udp_follower_tracking_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.udp_follower_tracking_btn.toggled.connect(self.on_udp_follower_tracking_toggled)
        tracking_layout.addWidget(self.udp_follower_tracking_btn)

        layout.addWidget(tracking_group)

        # --- 第三部分：单点遥操作移动 ---
        single_point_group = QGroupBox("单点遥操作移动 (UDP增量模式)")
        single_point_layout = QVBoxLayout(single_point_group)

        # 当前坐标显示
        current_pos_layout = QHBoxLayout()
        current_pos_layout.addWidget(QLabel("当前坐标:"))
        self.current_pos_label = QLabel("未获取")
        self.current_pos_label.setStyleSheet("color: #2ecc71; font-family: Courier New; font-weight: bold;")
        current_pos_layout.addWidget(self.current_pos_label)
        single_point_layout.addLayout(current_pos_layout)

        # 刷新当前坐标按钮
        self.refresh_pos_btn = QPushButton("🔄 获取当前坐标")
        self.refresh_pos_btn.setMinimumHeight(35)
        self.refresh_pos_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.refresh_pos_btn.clicked.connect(self.get_current_position_requested.emit)
        single_point_layout.addWidget(self.refresh_pos_btn)

        # 分隔线
        line = QLabel("=" * 50)
        line.setStyleSheet("color: #7f8c8d;")
        single_point_layout.addWidget(line)

        # 目标坐标输入区域
        single_point_layout.addWidget(QLabel("目标坐标 (绝对位置):"))
        input_grid = QGridLayout()
        
        # X, Y, Z 输入
        input_grid.addWidget(QLabel("X (mm):"), 0, 0)
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("目标X坐标(mm)")
        input_grid.addWidget(self.x_input, 0, 1)
        
        input_grid.addWidget(QLabel("Y (mm):"), 0, 2)
        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("目标Y坐标(mm)")
        input_grid.addWidget(self.y_input, 0, 3)
        
        input_grid.addWidget(QLabel("Z (mm):"), 1, 0)
        self.z_input = QLineEdit()
        self.z_input.setPlaceholderText("目标Z坐标(mm)")
        input_grid.addWidget(self.z_input, 1, 1)
        
        # Rx, Ry, Rz 输入
        input_grid.addWidget(QLabel("Rx (°):"), 1, 2)
        self.rx_input = QLineEdit()
        self.rx_input.setPlaceholderText("目标Rx角度(°)")
        input_grid.addWidget(self.rx_input, 1, 3)
        
        input_grid.addWidget(QLabel("Ry (°):"), 2, 0)
        self.ry_input = QLineEdit()
        self.ry_input.setPlaceholderText("目标Ry角度(°)")
        input_grid.addWidget(self.ry_input, 2, 1)
        
        input_grid.addWidget(QLabel("Rz (°):"), 2, 2)
        self.rz_input = QLineEdit()
        self.rz_input.setPlaceholderText("目标Rz角度(°)")
        input_grid.addWidget(self.rz_input, 2, 3)
        
        single_point_layout.addLayout(input_grid)

        # 提示标签
        self.single_point_info = QLabel("先获取当前坐标，再输入目标坐标\n系统会自动计算偏移量并移动")
        self.single_point_info.setWordWrap(True)
        self.single_point_info.setStyleSheet("color: #bdc3c7; font-size: 11px;")
        single_point_layout.addWidget(self.single_point_info)

        # 移动按钮
        self.single_point_move_btn = QPushButton("▶ 执行移动到目标坐标")
        self.single_point_move_btn.setMinimumHeight(50)
        self.single_point_move_btn.setStyleSheet("background-color: #16a085; color: white; font-weight: bold;")
        self.single_point_move_btn.clicked.connect(self.on_single_point_move_clicked)
        single_point_layout.addWidget(self.single_point_move_btn)

        layout.addWidget(single_point_group)

        # --- 第四部分：传送带白点检测显示 ---
        white_points_group = QGroupBox("传送带白点检测 (视野范围内)")
        white_points_layout = QVBoxLayout(white_points_group)

        # 白点数量显示
        self.white_points_count_label = QLabel("视野内白点数量: 0")
        self.white_points_count_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        white_points_layout.addWidget(self.white_points_count_label)

        # 白点位置列表
        self.white_points_text = QLabel("暂无白点进入视野")
        self.white_points_text.setWordWrap(True)
        self.white_points_text.setStyleSheet("color: #ecf0f1; font-family: Courier New; font-size: 11px;")
        self.white_points_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        white_points_layout.addWidget(self.white_points_text)

        # 启用/禁用白点检测
        self.white_points_toggle_btn = QPushButton("🔍 启用白点检测")
        self.white_points_toggle_btn.setCheckable(True)
        self.white_points_toggle_btn.setMinimumHeight(40)
        self.white_points_toggle_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.white_points_toggle_btn.toggled.connect(self.on_white_points_toggle)
        white_points_layout.addWidget(self.white_points_toggle_btn)

        layout.addWidget(white_points_group)

    def on_offset_tracking_toggled(self, checked):
        """处理自适应隐性 Offset 追踪按钮点击"""
        if checked:
            self.offset_tracking_btn.setText("停止传送带追踪 (自适应隐性Offset)")
            # 互斥取消其他模式
            if self.tracking_btn.isChecked():
                self.tracking_btn.setChecked(False)
            if self.sim_tracking_btn.isChecked():
                self.sim_tracking_btn.setChecked(False)
        else:
            self.offset_tracking_btn.setText("开启传送带追踪 (自适应隐性Offset)")
        self.conveyor_offset_tracking_toggled.emit(checked)

    def on_sim_tracking_toggled(self, checked):
        """处理仿真追踪按钮点击"""
        if checked:
            self.sim_tracking_btn.setText("停止传送带追踪 (仿真速度PI控制)")
            # 互斥取消其他模式
            if self.tracking_btn.isChecked():
                self.tracking_btn.setChecked(False)
            if hasattr(self, 'offset_tracking_btn') and self.offset_tracking_btn.isChecked():
                self.offset_tracking_btn.setChecked(False)
        else:
            self.sim_tracking_btn.setText("开启传送带追踪 (仿真速度PI控制)")
        self.conveyor_sim_tracking_toggled.emit(checked)

    def on_hover_only_toggled(self, checked):
        """处理仅悬停按钮点击"""
        if checked:
            self.hover_only_btn.setText("停止仅悬停追踪")
        else:
            self.hover_only_btn.setText("开启仅悬停追踪 (不触碰)")
        self.conveyor_hover_only_toggled.emit(checked)

    def on_tracking_toggled(self, checked):
        """处理追踪按钮点击"""
        if checked:
            self.tracking_btn.setText("停止传送带追踪")
        else:
            self.tracking_btn.setText("开启传送带追踪")
        
        # 发射信号给 MainWindow
        self.conveyor_tracking_toggled.emit(checked)

    def _on_speed_slider_changed(self, value):
        """处理速度滑块变化"""
        speed = value / 1000.0
        self.speed_val_label.setText(f"{speed:.3f} m/s")
        self.conveyor_speed_changed.emit(speed)

    def on_load_clicked(self):
        """打开文件对话框选择并加载 CSV 轨迹"""
        # 设置默认打开目录为项目下的 csv 文件夹
        default_dir = os.path.join(os.getcwd(), 'csv')
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视觉轨迹 CSV 文件", default_dir, "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            # 尝试通过列名匹配，如果失败则按顺序取前6列
            # 转换为小写以增加兼容性
            cols = [c.lower() for c in df.columns]
            
            if all(k in cols for k in ['x', 'y', 'z']):
                # 按名称提取
                target_cols = []
                for k in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
                    # 寻找匹配的原始列名
                    match = [c for c in df.columns if c.lower() == k]
                    if match:
                        target_cols.append(match[0])
                    elif k in ['rx', 'ry', 'rz']:
                        # 如果没有旋转列，补充默认值 (Rx=180 为末端向下)
                        df[k] = 180.0 if k == 'rx' else 0.0
                        target_cols.append(k)
                
                points = df[target_cols].values.tolist()
            else:
                # 兼容旧版：如果没有 x,y,z 列名，则按顺序取前 6 列
                points = df.iloc[:, :6].values.tolist()
            
            if len(points) > 0:
                self.csv_path = file_path # 更新当前使用的路径
                self.trajectory_generated.emit(points)
                self.execute_btn.setEnabled(True)
                self.execute_udp_btn.setEnabled(True)
                filename = os.path.basename(file_path)
                self.result_label.setText(f"已加载: {filename}\n({len(points)} 个点)")
            else:
                self.result_label.setText("错误：CSV 文件为空")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取 CSV 失败: {str(e)}")

    def on_udp_follower_tracking_toggled(self, checked):
        """处理 UDP follower_cart 追踪按钮点击"""
        self.udp_follower_tracking_btn.setText("停止UDP追踪" if checked else "UDP追踪小球")
        self.conveyor_udp_follower_tracking_toggled.emit(checked)

    def on_udp_execution_clicked(self):
        """点击 UDP 增量执行"""
        self.udp_execution_requested.emit()

    def on_single_point_move_clicked(self):
        """处理单点遥操作移动按钮点击"""
        try:
            # 检查是否已获取当前坐标
            if self.current_pos_label.text() == "未获取":
                QMessageBox.warning(self, "提示", "请先点击'获取当前坐标'按钮")
                return
            
            # 获取输入值
            if not self.x_input.text().strip():
                QMessageBox.warning(self, "输入错误", "请输入目标X坐标")
                return
            if not self.y_input.text().strip():
                QMessageBox.warning(self, "输入错误", "请输入目标Y坐标")
                return
            if not self.z_input.text().strip():
                QMessageBox.warning(self, "输入错误", "请输入目标Z坐标")
                return
                
            x = float(self.x_input.text().strip())
            y = float(self.y_input.text().strip())
            z = float(self.z_input.text().strip())
            rx = float(self.rx_input.text().strip()) if self.rx_input.text().strip() else 180.0
            ry = float(self.ry_input.text().strip()) if self.ry_input.text().strip() else 0.0
            rz = float(self.rz_input.text().strip()) if self.rz_input.text().strip() else 0.0
            
            self.single_point_move_requested.emit(x, y, z, rx, ry, rz)
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"请输入有效的数字: {e}")
    
    def update_current_position_display(self, x, y, z, rx, ry, rz):
        """更新当前坐标显示"""
        self.current_pos_label.setText(f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f} Rx:{rx:.2f} Ry:{ry:.2f} Rz:{rz:.2f}")
        
        # 自动填充目标坐标输入框为当前坐标（方便用户微调）
        self.x_input.setText(f"{x:.2f}")
        self.y_input.setText(f"{y:.2f}")
        self.z_input.setText(f"{z:.2f}")
        self.rx_input.setText(f"{rx:.2f}")
        self.ry_input.setText(f"{ry:.2f}")
        self.rz_input.setText(f"{rz:.2f}")
    
    def on_white_points_toggle(self, checked):
        """处理白点检测按钮状态改变"""
        if checked:
            self.white_points_toggle_btn.setText("🔴 禁用白点检测")
        else:
            self.white_points_toggle_btn.setText("🔍 启用白点检测")
            # 清空显示
            self.white_points_count_label.setText("视野内: 0个")
            self.white_points_text.setText("检测已禁用")
    
    def update_white_points_display(self, points_list):
        """
        更新视野内白点/绿点位置显示
        
        Args:
            points_list: [(x, y, z, id, state), ...] 点位置列表（单位：米）
                        state: 0=白点(WAITING), 1=绿点(TRACKING)
        """
        count = len(points_list)
        white_count = sum(1 for p in points_list if p[4] == 0)
        green_count = sum(1 for p in points_list if p[4] == 1)
        
        self.white_points_count_label.setText(f"视野内: {count}个 (白:{white_count} 绿:{green_count})")
        
        if count == 0:
            self.white_points_text.setText("暂无白点进入视野")
        else:
            text_lines = ["点位置 (X, Y, Z mm):"]
            for x, y, z, pid, state in points_list:
                # 根据状态选择颜色标签
                status_label = "🟢追踪" if state == 1 else "⚪等待"
                # 转换为毫米并格式化
                text_lines.append(f"  ID{pid} {status_label}: X={x*1000:6.1f} Y={y*1000:6.1f} Z={z*1000:6.1f}")
            self.white_points_text.setText("\n".join(text_lines))
