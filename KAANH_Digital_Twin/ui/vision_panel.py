"""
视觉功能面板 (CSV 文件版)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QMessageBox, QFileDialog,
    QHBoxLayout, QSlider
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
    conveyor_speed_changed = pyqtSignal(float)
    conveyor_hover_only_toggled = pyqtSignal(bool)
    
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

        self.tracking_btn = QPushButton("开启传送带追踪")
        self.tracking_btn.setCheckable(True)
        self.tracking_btn.setMinimumHeight(50)
        self.tracking_btn.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #e74c3c; }
        """)
        self.tracking_btn.toggled.connect(self.on_tracking_toggled)
        tracking_layout.addWidget(self.tracking_btn)

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
            QPushButton { background-color: #8e44ad; color: white; font-weight: bold; }
            QPushButton:checked { background-color: #9b59b6; }
        """)
        self.hover_only_btn.toggled.connect(self.on_hover_only_toggled)
        tracking_layout.addWidget(self.hover_only_btn)

        layout.addWidget(tracking_group)

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

    def on_udp_execution_clicked(self):
        """点击 UDP 增量执行"""
        self.udp_execution_requested.emit()
