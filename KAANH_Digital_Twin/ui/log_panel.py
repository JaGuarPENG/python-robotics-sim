"""
日志显示面板 - 捕获并显示终端打印信息
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QPlainTextEdit
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

class LogPanel(QWidget):
    """日志面板：显示系统运行过程中的终端输出"""
    
    def __init__(self, signals, parent=None):
        super().__init__(parent)
        self.signals = signals
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("运行日志 (System Logs)")
        v_layout = QVBoxLayout(group)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        # 设置较小的等宽字体
        self.log_output.setFont(QFont("Consolas", 14))
        self.log_output.setStyleSheet("""
            background-color: #000000;
            color: #2ecc71;
            border: none;
        """)
        # 设置最大行数，防止内存占用过大
        self.log_output.setMaximumBlockCount(500)
        
        v_layout.addWidget(self.log_output)
        layout.addWidget(group)

    def connect_signals(self):
        self.signals.log_message.connect(self.append_log)

    @pyqtSlot(str)
    def append_log(self, message):
        """添加一条日志"""
        self.log_output.appendPlainText(message)
        # 自动滚动到底部
        self.log_output.ensureCursorVisible()
