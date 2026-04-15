# -*- coding: utf-8 -*-
"""
示教器面板 - 只嵌入第4个窗口并自适应大小
"""

import os
import subprocess
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QMessageBox, QLineEdit, QWidget
)
from PyQt5.QtCore import QTimer

try:
    import win32gui
    import win32con
    import win32process
    WINDOWS_API = True
except:
    WINDOWS_API = False


class TeachPanel(QGroupBox):
    DEFAULT_PATH = r"D:\\project\\Visual_servoing\\code\\KaPI-win-2.5.7\\KaanhTeacher-2.5.7.exe"
    
    def __init__(self, parent=None):
        super().__init__("外部示教器", parent)
        self.process = None
        self.pid = None
        self.target_hwnd = None
        self.window_list = []
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_windows)
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("程序路径:"))
        self.path_input = QLineEdit(self.DEFAULT_PATH)
        path_layout.addWidget(self.path_input)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.on_browse)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        
        btn_layout = QHBoxLayout()
        self.launch_btn = QPushButton("启动示教器")
        self.launch_btn.clicked.connect(self.on_launch)
        btn_layout.addWidget(self.launch_btn)
        self.stop_btn = QPushButton("关闭示教器")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        self.status_label = QLabel("状态: 未启动")
        layout.addWidget(self.status_label)
        
        self.container = QWidget()
        self.container.setMinimumSize(1126, 768)
        self.container.setStyleSheet("background-color: #2c3e50; border: 3px solid #3498db;")
        layout.addWidget(self.container)
        
        layout.addWidget(QLabel("提示: 自动获取第4个窗口并嵌入"))
        
    def on_browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择程序", "", "*.exe")
        if f:
            self.path_input.setText(f)
    
    def find_windows(self):
        if not WINDOWS_API:
            return []
        result = []
        def cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                try:
                    title = win32gui.GetWindowText(hwnd)
                    if "kaanh" in title.lower():
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid == self.pid:
                            result.append((hwnd, title))
                except:
                    pass
            return True
        win32gui.EnumWindows(cb, None)
        return result
    
    def embed_and_resize(self, hwnd):
        if not WINDOWS_API:
            return False
        try:
            container = int(self.container.winId())
            win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_SYSMENU | 
                      win32con.WS_POPUP | win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX)
            style |= win32con.WS_CHILD | win32con.WS_CLIPCHILDREN
            win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
            
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            ex_style &= ~(win32con.WS_EX_WINDOWEDGE | win32con.WS_EX_CLIENTEDGE | 
                         win32con.WS_EX_DLGMODALFRAME | win32con.WS_EX_TOOLWINDOW)
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)
            
            win32gui.SetParent(hwnd, container)
            self.resize_to_fit(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            return True
        except Exception as e:
            print(f"Embed error: {e}")
            return False
    
    def resize_to_fit(self, hwnd=None):
        if not WINDOWS_API:
            return
        hwnd = hwnd or self.target_hwnd
        if not hwnd:
            return
        try:
            geo = self.container.geometry()
            w, h = geo.width(), geo.height()
            win32gui.SetWindowPos(
                hwnd, win32con.HWND_TOP,
                0, 0, w, h,
                win32con.SWP_FRAMECHANGED | win32con.SWP_NOZORDER
            )
        except Exception as e:
            print(f"Resize error: {e}")
    
    def on_launch(self):
        exe = self.path_input.text().strip()
        if not os.path.exists(exe):
            QMessageBox.warning(self, "警告", "程序不存在")
            return
        try:
            exe_dir = os.path.dirname(exe)
            original = os.getcwd()
            os.chdir(exe_dir)
            self.process = subprocess.Popen(exe, stdout=subprocess.DEVNULL, 
                                            stderr=subprocess.DEVNULL)
            os.chdir(original)
            self.pid = self.process.pid
            self.window_list = []
            self.target_hwnd = None
            self.status_label.setText("状态: 已启动，等待第4个窗口...")
            self.timer.start(500)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
    
    def check_windows(self):
        if not self.process or self.process.poll() is not None:
            self.timer.stop()
            return
        
        if self.target_hwnd:
            self.resize_to_fit()
            return
        
        windows = self.find_windows()
        for hwnd, title in windows:
            if hwnd not in [h for h, t in self.window_list]:
                self.window_list.append((hwnd, title))
                count = len(self.window_list)
                self.status_label.setText(f"状态: 发现第{count}个窗口")
                
                if count == 4:
                    if self.embed_and_resize(hwnd):
                        self.target_hwnd = hwnd
                        self.status_label.setText("状态: 已嵌入第4个窗口")
                        self.launch_btn.setEnabled(False)
                        self.stop_btn.setEnabled(True)
    
    def on_stop(self):
        self.timer.stop()
        if self.target_hwnd:
            try:
                win32gui.SetParent(self.target_hwnd, 0)
            except:
                pass
        if self.process:
            try:
                self.process.terminate()
                self.process.wait()
            except:
                pass
        self.target_hwnd = None
        self.window_list = []
        self.process = None
        self.status_label.setText("状态: 已关闭")
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.target_hwnd:
            self.resize_to_fit()
    
    def hideEvent(self, event):
        super().hideEvent(event)
        if self.target_hwnd and WINDOWS_API:
            try:
                win32gui.ShowWindow(self.target_hwnd, win32con.SW_HIDE)
            except:
                pass
    
    def showEvent(self, event):
        super().showEvent(event)
        if self.target_hwnd and WINDOWS_API:
            try:
                win32gui.ShowWindow(self.target_hwnd, win32con.SW_SHOW)
            except:
                pass
