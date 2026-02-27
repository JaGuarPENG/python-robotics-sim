"""
视觉功能面板 (CSV 文件版)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QMessageBox, QFileDialog
)
from PyQt5.QtCore import pyqtSignal, Qt
import pandas as pd
import os

class VisionPanel(QWidget):
    """视觉功能面板：加载 CSV 轨迹并执行"""
    
    trajectory_generated = pyqtSignal(list)
    execution_requested = pyqtSignal()
    udp_execution_requested = pyqtSignal()
    actual_export_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_path = 'csv/vision_trajectory.csv'
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("视觉引导 (CSV 导入模式)")
        v_layout = QVBoxLayout(group)

        self.info_label = QLabel("1. 运行 verify_vision.py 并保存 CSV\n2. 点击下方按钮导入并执行")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #bdc3c7; font-size: 11px;")
        v_layout.addWidget(self.info_label)

        # 按钮区
        self.load_btn = QPushButton("第一步：加载视觉 CSV 轨迹")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.load_btn.clicked.connect(self.on_load_clicked)
        v_layout.addWidget(self.load_btn)

        self.execute_btn = QPushButton("第二步：执行运动 (IK 模式)")
        self.execute_btn.setMinimumHeight(40)
        self.execute_btn.setEnabled(False)
        self.execute_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        self.execute_btn.clicked.connect(self.execution_requested.emit)
        v_layout.addWidget(self.execute_btn)

        self.execute_udp_btn = QPushButton("第二步：执行运动 (UDP 增量模式)")
        self.execute_udp_btn.setMinimumHeight(40)
        self.execute_udp_btn.setEnabled(False)
        self.execute_udp_btn.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
        self.execute_udp_btn.clicked.connect(self.on_udp_execution_clicked)
        v_layout.addWidget(self.execute_udp_btn)

        self.export_btn = QPushButton("第三步：导出实际运行 CSV")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.export_btn.clicked.connect(self.actual_export_requested.emit)
        v_layout.addWidget(self.export_btn)
        
        # 结果显示
        self.result_label = QLabel("等待导入...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-family: Courier New; color: #f1c40f; background: #2c3e50; padding: 5px; border-radius: 3px;")
        v_layout.addWidget(self.result_label)

        layout.addWidget(group)

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
