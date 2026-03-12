"""
视觉功能面板 (优化版 - 使用子标签页组织)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QMessageBox, QFileDialog,
    QHBoxLayout, QSlider, QLineEdit, QGridLayout, QTabWidget
)
from PyQt5.QtCore import pyqtSignal, Qt
import pandas as pd
import os

class VisionPanel(QWidget):
    """视觉功能面板：CSV轨迹、传送带控制、遥操作"""
    
    trajectory_generated = pyqtSignal(list)
    execution_requested = pyqtSignal()
    udp_execution_requested = pyqtSignal()
    actual_export_requested = pyqtSignal()
    
    # 传送带追踪信号
    conveyor_tracking_toggled = pyqtSignal(bool)
    conveyor_sim_tracking_toggled = pyqtSignal(bool)
    conveyor_offset_tracking_toggled = pyqtSignal(bool)
    conveyor_speed_changed = pyqtSignal(float)
    conveyor_hover_only_toggled = pyqtSignal(bool)
    conveyor_udp_follower_tracking_toggled = pyqtSignal(bool)
    
    # 单点遥操作信号
    single_point_move_requested = pyqtSignal(float, float, float, float, float, float)
    get_current_position_requested = pyqtSignal()
    
    # UDP+前馈测试信号
    udp_feedforward_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_path = 'csv/vision_trajectory.csv'
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 使用子标签页组织功能
        self.tabs = QTabWidget()
        
        # Tab 1: CSV轨迹
        self.tabs.addTab(self._create_csv_tab(), "📁 CSV轨迹")
        
        # Tab 2: 传送带控制
        self.tabs.addTab(self._create_conveyor_tab(), "🔄 传送带")
        
        # Tab 3: 遥操作
        self.tabs.addTab(self._create_teleop_tab(), "🎮 遥操作")
        
        layout.addWidget(self.tabs)

    def _create_csv_tab(self):
        """创建CSV轨迹导入标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 说明
        info = QLabel("1. 运行 verify_vision.py 保存 CSV\n2. 点击下方按钮导入并执行")
        info.setWordWrap(True)
        info.setStyleSheet("color: #bdc3c7; font-size: 11px;")
        layout.addWidget(info)

        # 加载按钮
        self.load_btn = QPushButton("📂 加载视觉CSV轨迹")
        self.load_btn.setMinimumHeight(35)
        self.load_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.load_btn.clicked.connect(self.on_load_clicked)
        layout.addWidget(self.load_btn)

        # 执行按钮组
        btn_layout = QHBoxLayout()
        
        self.execute_btn = QPushButton("执行(IK)")
        self.execute_btn.setEnabled(False)
        self.execute_btn.setStyleSheet("background-color: #c0392b; color: white;")
        self.execute_btn.clicked.connect(self.execution_requested.emit)
        btn_layout.addWidget(self.execute_btn)

        self.execute_udp_btn = QPushButton("执行(UDP)")
        self.execute_udp_btn.setEnabled(False)
        self.execute_udp_btn.setStyleSheet("background-color: #d35400; color: white;")
        self.execute_udp_btn.clicked.connect(self.on_udp_execution_clicked)
        btn_layout.addWidget(self.execute_udp_btn)
        
        layout.addLayout(btn_layout)

        # 导出按钮
        self.export_btn = QPushButton("📥 导出实际轨迹CSV")
        self.export_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.export_btn.clicked.connect(self.actual_export_requested.emit)
        layout.addWidget(self.export_btn)
        
        # 结果显示
        self.result_label = QLabel("等待导入...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-family: Courier New; color: #f1c40f; background: #2c3e50; "
            "padding: 8px; border-radius: 3px; font-size: 12px;"
        )
        layout.addWidget(self.result_label)
        
        layout.addStretch()
        return tab

    def _create_conveyor_tab(self):
        """创建传送带控制标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 白点检测（放在最上面，重要功能）
        detect_group = QGroupBox("👁 白点检测")
        detect_layout = QVBoxLayout(detect_group)
        detect_layout.setContentsMargins(5, 5, 5, 5)
        
        self.white_points_count_label = QLabel("视野内: 0个")
        self.white_points_count_label.setStyleSheet(
            "color: #f39c12; font-weight: bold; font-size: 28px;"
        )
        detect_layout.addWidget(self.white_points_count_label)

        self.white_points_text = QLabel("暂无白点")
        self.white_points_text.setWordWrap(True)
        self.white_points_text.setStyleSheet(
            "color: #ecf0f1; font-family: Courier New; font-size: 24px;"
        )
        self.white_points_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.white_points_text.setMinimumHeight(120)
        detect_layout.addWidget(self.white_points_text)

        self.white_points_toggle_btn = QPushButton("🔍 启用检测")
        self.white_points_toggle_btn.setCheckable(True)
        self.white_points_toggle_btn.setMinimumHeight(30)
        self.white_points_toggle_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.white_points_toggle_btn.toggled.connect(self.on_white_points_toggle)
        detect_layout.addWidget(self.white_points_toggle_btn)
        layout.addWidget(detect_group)

        # 速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 200)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_val_label = QLabel("0.05 m/s")
        self.speed_val_label.setFixedWidth(80)
        speed_layout.addWidget(self.speed_val_label)
        layout.addLayout(speed_layout)

        # 追踪模式 - 只保留自适应和UDP追踪
        tracking_layout = QVBoxLayout()
        tracking_layout.setSpacing(5)
        
        tracking_label = QLabel("追踪模式:")
        tracking_label.setStyleSheet("color: #bdc3c7; font-size: 12px;")
        tracking_layout.addWidget(tracking_label)
        
        # 自适应追踪按钮
        self.offset_tracking_btn = QPushButton("🎯 自适应追踪")
        self.offset_tracking_btn.setCheckable(True)
        self.offset_tracking_btn.setMinimumHeight(45)
        self.offset_tracking_btn.setStyleSheet("""
            QPushButton { background-color: #9b59b6; color: white; font-weight: bold; font-size: 14px; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.offset_tracking_btn.toggled.connect(self.on_offset_tracking_toggled)
        tracking_layout.addWidget(self.offset_tracking_btn)
        
        # UDP追踪按钮（纯UDP，无前馈）
        self.udp_follower_tracking_btn = QPushButton("📡 UDP追踪")
        self.udp_follower_tracking_btn.setCheckable(True)
        self.udp_follower_tracking_btn.setMinimumHeight(45)
        self.udp_follower_tracking_btn.setStyleSheet("""
            QPushButton { background-color: #e67e22; color: white; font-weight: bold; font-size: 14px; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.udp_follower_tracking_btn.toggled.connect(self.on_udp_follower_tracking_toggled)
        tracking_layout.addWidget(self.udp_follower_tracking_btn)
        
        # UDP+前馈测试按钮（实验性）
        self.udp_feedforward_btn = QPushButton("🧪 UDP+前馈(测试)")
        self.udp_feedforward_btn.setCheckable(True)
        self.udp_feedforward_btn.setMinimumHeight(45)
        self.udp_feedforward_btn.setStyleSheet("""
            QPushButton { background-color: #16a085; color: white; font-weight: bold; font-size: 14px; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.udp_feedforward_btn.toggled.connect(self.on_udp_feedforward_toggled)
        tracking_layout.addWidget(self.udp_feedforward_btn)
        
        layout.addLayout(tracking_layout)
        layout.addStretch()
        return tab

    def _create_teleop_tab(self):
        """创建遥操作标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 当前坐标显示
        current_group = QGroupBox("📍 当前坐标")
        current_layout = QVBoxLayout(current_group)
        current_layout.setContentsMargins(5, 5, 5, 5)
        
        self.current_pos_label = QLabel("未获取")
        self.current_pos_label.setStyleSheet(
            "color: #2ecc71; font-family: Courier New; font-weight: bold; font-size: 11px;"
        )
        current_layout.addWidget(self.current_pos_label)

        self.refresh_pos_btn = QPushButton("🔄 获取当前坐标")
        self.refresh_pos_btn.setMinimumHeight(30)
        self.refresh_pos_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.refresh_pos_btn.clicked.connect(self.get_current_position_requested.emit)
        current_layout.addWidget(self.refresh_pos_btn)
        layout.addWidget(current_group)

        # 目标坐标输入
        target_group = QGroupBox("🎯 目标坐标")
        target_layout = QVBoxLayout(target_group)
        target_layout.setContentsMargins(5, 5, 5, 5)
        
        # 紧凑的输入网格
        input_grid = QGridLayout()
        input_grid.setSpacing(3)
        
        # X, Y, Z
        input_grid.addWidget(QLabel("X:"), 0, 0)
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("mm")
        input_grid.addWidget(self.x_input, 0, 1)
        
        input_grid.addWidget(QLabel("Y:"), 0, 2)
        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("mm")
        input_grid.addWidget(self.y_input, 0, 3)
        
        input_grid.addWidget(QLabel("Z:"), 1, 0)
        self.z_input = QLineEdit()
        self.z_input.setPlaceholderText("mm")
        input_grid.addWidget(self.z_input, 1, 1)
        
        input_grid.addWidget(QLabel("Rx:"), 1, 2)
        self.rx_input = QLineEdit()
        self.rx_input.setPlaceholderText("°")
        input_grid.addWidget(self.rx_input, 1, 3)
        
        input_grid.addWidget(QLabel("Ry:"), 2, 0)
        self.ry_input = QLineEdit()
        self.ry_input.setPlaceholderText("°")
        input_grid.addWidget(self.ry_input, 2, 1)
        
        input_grid.addWidget(QLabel("Rz:"), 2, 2)
        self.rz_input = QLineEdit()
        self.rz_input.setPlaceholderText("°")
        input_grid.addWidget(self.rz_input, 2, 3)
        
        target_layout.addLayout(input_grid)

        # 移动按钮
        self.single_point_move_btn = QPushButton("▶ 执行移动")
        self.single_point_move_btn.setMinimumHeight(40)
        self.single_point_move_btn.setStyleSheet(
            "background-color: #16a085; color: white; font-weight: bold;"
        )
        self.single_point_move_btn.clicked.connect(self.on_single_point_move_clicked)
        target_layout.addWidget(self.single_point_move_btn)
        
        layout.addWidget(target_group)
        layout.addStretch()
        return tab

    # ==================== 事件处理 ====================

    def _on_speed_slider_changed(self, value):
        """处理速度滑块变化"""
        speed = value / 1000.0
        self.speed_val_label.setText(f"{speed:.3f} m/s")
        self.conveyor_speed_changed.emit(speed)

    def on_load_clicked(self):
        """打开文件对话框选择并加载 CSV 轨迹"""
        default_dir = os.path.join(os.getcwd(), 'csv')
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视觉轨迹 CSV 文件", default_dir, 
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            cols = [c.lower() for c in df.columns]
            
            if all(k in cols for k in ['x', 'y', 'z']):
                target_cols = []
                for k in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
                    match = [c for c in df.columns if c.lower() == k]
                    if match:
                        target_cols.append(match[0])
                    elif k in ['rx', 'ry', 'rz']:
                        df[k] = 180.0 if k == 'rx' else 0.0
                        target_cols.append(k)
                points = df[target_cols].values.tolist()
            else:
                points = df.iloc[:, :6].values.tolist()
            
            if len(points) > 0:
                self.csv_path = file_path
                self.trajectory_generated.emit(points)
                self.execute_btn.setEnabled(True)
                self.execute_udp_btn.setEnabled(True)
                filename = os.path.basename(file_path)
                self.result_label.setText(f"已加载: {filename}\n({len(points)} 个点)")
            else:
                self.result_label.setText("错误：CSV 文件为空")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取 CSV 失败: {str(e)}")

    def on_udp_execution_clicked(self):
        """点击 UDP 增量执行"""
        self.udp_execution_requested.emit()

    def on_single_point_move_clicked(self):
        """处理单点遥操作移动按钮点击"""
        try:
            if self.current_pos_label.text() == "未获取":
                QMessageBox.warning(self, "提示", "请先点击'获取当前坐标'按钮")
                return
            
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
        self.current_pos_label.setText(
            f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}\nRx:{rx:.1f} Ry:{ry:.1f} Rz:{rz:.1f}"
        )
        self.x_input.setText(f"{x:.2f}")
        self.y_input.setText(f"{y:.2f}")
        self.z_input.setText(f"{z:.2f}")
        self.rx_input.setText(f"{rx:.2f}")
        self.ry_input.setText(f"{ry:.2f}")
        self.rz_input.setText(f"{rz:.2f}")

    def on_white_points_toggle(self, checked):
        """处理白点检测按钮状态改变"""
        if checked:
            self.white_points_toggle_btn.setText("🔴 禁用检测")
        else:
            self.white_points_toggle_btn.setText("🔍 启用检测")
            self.white_points_count_label.setText("视野内: 0个")
            self.white_points_text.setText("检测已禁用")

    def update_white_points_display(self, points_list):
        """更新视野内白点/绿点位置显示"""
        count = len(points_list)
        white_count = sum(1 for p in points_list if p[4] == 0)
        green_count = sum(1 for p in points_list if p[4] == 1)
        
        self.white_points_count_label.setText(f"视野内: {count}个 (白:{white_count} 绿:{green_count})")
        
        if count == 0:
            self.white_points_text.setText("暂无白点")
        else:
            lines = []
            for x, y, z, pid, state in points_list:
                status = "🟢" if state == 1 else "⚪"
                lines.append(f"{status}ID{pid}: {x*1000:.0f},{y*1000:.0f},{z*1000:.0f}")
            self.white_points_text.setText("\n".join(lines[:5]))  # 最多显示5个

    # 追踪按钮处理
    def on_offset_tracking_toggled(self, checked):
        self.offset_tracking_btn.setText("🛑 停止自适应" if checked else "🎯 自适应追踪")
        self.conveyor_offset_tracking_toggled.emit(checked)

    def on_udp_follower_tracking_toggled(self, checked):
        self.udp_follower_tracking_btn.setText("🛑 停止UDP" if checked else "📡 UDP追踪")
        self.conveyor_udp_follower_tracking_toggled.emit(checked)
        
    def on_udp_feedforward_toggled(self, checked):
        self.udp_feedforward_btn.setText("🛑 停止前馈测试" if checked else "🧪 UDP+前馈(测试)")
        self.udp_feedforward_toggled.emit(checked)
