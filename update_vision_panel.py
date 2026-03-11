import os
filepath = r'KAANH_Digital_Twin/ui/vision_panel.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

import re
old_init_ui = re.search(r'    def init_ui\(self\):.*?        layout\.addWidget\(tracking_group\)', content, re.DOTALL)
if old_init_ui:
    new_btn_code = '''
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
'''
    new_init_ui = old_init_ui.group(0).replace('tracking_layout.addLayout(speed_layout)', 'tracking_layout.addLayout(speed_layout)
' + new_btn_code)
    content = content.replace(old_init_ui.group(0), new_init_ui)
    
    # Add new signal
    content = content.replace('conveyor_tracking_toggled = pyqtSignal(bool)', 'conveyor_tracking_toggled = pyqtSignal(bool)
    conveyor_hover_only_toggled = pyqtSignal(bool)')
    
    # Add new slot
    new_slot = '''    def on_hover_only_toggled(self, checked):
        """处理仅悬停按钮点击"""
        if checked:
            self.hover_only_btn.setText("停止仅悬停追踪")
        else:
            self.hover_only_btn.setText("开启仅悬停追踪 (不触碰)")
        self.conveyor_hover_only_toggled.emit(checked)
'''
    content = content.replace('    def _on_speed_slider_changed(self, value):', new_slot + '
    def _on_speed_slider_changed(self, value):')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated vision_panel.py')
else:
    print('Could not find init_ui')
