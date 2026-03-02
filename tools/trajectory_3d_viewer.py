#!/usr/bin/env python3
"""
3D轨迹可视化工具 - 用于显示CSV文件中的机器人轨迹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.colorchooser import askcolor
import os
import glob
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """3D箭头类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


class Trajectory3DViewer:
    """3D轨迹查看器主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("3D轨迹可视化工具")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # 数据存储
        self.trajectories = {}  # {name: {'data': df, 'color': color, 'visible': bool, 'show_orientation': bool}}
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                       '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        self.color_index = 0
        
        # 创建UI
        self._create_ui()
        
        # 加载默认CSV文件
        self._load_default_csv_files()
    
    def _create_ui(self):
        """创建用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 左侧控制面板
        self._create_control_panel(main_frame)
        
        # 右侧3D绘图区域
        self._create_plot_area(main_frame)
    
    def _create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        row = 0
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="5")
        file_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        ttk.Button(file_frame, text="添加CSV文件", command=self._add_csv_file).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(file_frame, text="刷新默认文件", command=self._load_default_csv_files).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(file_frame, text="清除所有", command=self._clear_all).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        row += 1
        
        # 轨迹列表区域
        list_frame = ttk.LabelFrame(control_frame, text="轨迹列表", padding="5")
        list_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # 创建Canvas和滚动条
        self.traj_canvas = tk.Canvas(list_frame, width=250, height=300)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.traj_canvas.yview)
        self.traj_list_frame = ttk.Frame(self.traj_canvas)
        
        self.traj_list_frame.bind(
            "<Configure>",
            lambda e: self.traj_canvas.configure(scrollregion=self.traj_canvas.bbox("all"))
        )
        
        self.traj_canvas.create_window((0, 0), window=self.traj_list_frame, anchor="nw")
        self.traj_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.traj_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        row += 1
        
        # 显示选项
        options_frame = ttk.LabelFrame(control_frame, text="显示选项", padding="5")
        options_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(0, weight=1)
        
        self.show_points_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="显示数据点", variable=self.show_points_var, 
                       command=self._update_plot).grid(row=0, column=0, sticky=tk.W)
        
        self.show_orientation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="显示姿态箭头", variable=self.show_orientation_var, 
                       command=self._update_plot).grid(row=1, column=0, sticky=tk.W)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="显示网格", variable=self.show_grid_var, 
                       command=self._update_plot).grid(row=2, column=0, sticky=tk.W)
        
        self.equal_aspect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="等比例坐标轴", variable=self.equal_aspect_var, 
                       command=self._update_plot).grid(row=3, column=0, sticky=tk.W)
        
        row += 1
        
        # 视角控制
        view_frame = ttk.LabelFrame(control_frame, text="视角控制", padding="5")
        view_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        view_frame.columnconfigure(0, weight=1)
        
        ttk.Button(view_frame, text="正视图 (XY平面)", command=lambda: self._set_view(0, 0, 90)).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=1)
        ttk.Button(view_frame, text="俯视图 (XZ平面)", command=lambda: self._set_view(0, 0, 0)).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=1)
        ttk.Button(view_frame, text="侧视图 (YZ平面)", command=lambda: self._set_view(0, 90, 90)).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=1)
        ttk.Button(view_frame, text="等轴测图", command=lambda: self._set_view(30, -45, 0)).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=1)
        ttk.Button(view_frame, text="重置视角", command=lambda: self._set_view(30, -60, 0)).grid(row=4, column=0, sticky=(tk.W, tk.E), pady=1)
        
        row += 1
        
        # 信息显示
        info_frame = ttk.LabelFrame(control_frame, text="轨迹信息", padding="5")
        info_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)
        
        self.info_text = tk.Text(info_frame, width=30, height=10, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text.config(state=tk.DISABLED)
        
        row += 1
        
        # 退出按钮
        ttk.Button(control_frame, text="退出", command=self.root.quit).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 使信息区域可扩展
        control_frame.rowconfigure(row-1, weight=1)
    
    def _create_plot_area(self, parent):
        """创建右侧3D绘图区域"""
        plot_frame = ttk.Frame(parent)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # 创建matplotlib图形
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置初始视角
        self.ax.view_init(elev=30, azim=-60)
        
        # 设置标签
        self.ax.set_xlabel('X (mm)', fontsize=10)
        self.ax.set_ylabel('Y (mm)', fontsize=10)
        self.ax.set_zlabel('Z (mm)', fontsize=10)
        self.ax.set_title('3D轨迹可视化', fontsize=12, fontweight='bold')
        
        # 创建Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # 添加保存按钮
        ttk.Button(toolbar_frame, text="保存图片", command=self._save_figure).pack(side=tk.RIGHT, padx=5)
    
    def _load_default_csv_files(self):
        """加载默认的CSV文件"""
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv")
        if os.path.exists(csv_dir):
            csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
            for csv_file in sorted(csv_files):
                self._load_csv_file(csv_file)
        self._update_trajectory_list()
        self._update_plot()
    
    def _add_csv_file(self):
        """添加CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self._load_csv_file(file_path)
            self._update_trajectory_list()
            self._update_plot()
    
    def _load_csv_file(self, file_path):
        """加载单个CSV文件"""
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'latin1', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                messagebox.showerror("错误", f"无法读取文件编码: {file_path}")
                return
            
            # 标准化列名（去除BOM标记和空格）
            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
            
            # 检查必要的列
            required_cols = ['x', 'y', 'z']
            col_mapping = {}
            for req_col in required_cols:
                found = False
                for col in df.columns:
                    if col.lower().replace(' ', '') == req_col:
                        col_mapping[col] = req_col
                        found = True
                        break
                if not found:
                    messagebox.showerror("错误", f"文件缺少必要的列 '{req_col}': {os.path.basename(file_path)}")
                    return
            
            # 重命名列
            if col_mapping:
                df = df.rename(columns=col_mapping)
            
            # 获取文件名
            file_name = os.path.basename(file_path)
            
            # 如果同名文件已存在，添加序号
            base_name = file_name
            counter = 1
            while file_name in self.trajectories:
                name, ext = os.path.splitext(base_name)
                file_name = f"{name}_{counter}{ext}"
                counter += 1
            
            # 存储轨迹数据
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            
            self.trajectories[file_name] = {
                'data': df,
                'color': color,
                'visible': True,
                'path': file_path
            }
            
            print(f"已加载: {file_name} ({len(df)} 个点)")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
    
    def _update_trajectory_list(self):
        """更新轨迹列表UI"""
        # 清除现有控件
        for widget in self.traj_list_frame.winfo_children():
            widget.destroy()
        
        # 添加每个轨迹的控件
        for i, (name, traj_data) in enumerate(self.trajectories.items()):
            traj_frame = ttk.Frame(self.traj_list_frame)
            traj_frame.pack(fill=tk.X, pady=2)
            
            # 可见性复选框
            visible_var = tk.BooleanVar(value=traj_data['visible'])
            cb = ttk.Checkbutton(traj_frame, variable=visible_var, 
                                command=lambda n=name, v=visible_var: self._toggle_visibility(n, v))
            cb.pack(side=tk.LEFT)
            
            # 颜色选择按钮
            color_btn = tk.Button(traj_frame, bg=traj_data['color'], width=2, 
                                 command=lambda n=name: self._change_color(n))
            color_btn.pack(side=tk.LEFT, padx=2)
            
            # 文件名标签
            name_label = ttk.Label(traj_frame, text=name, width=25)
            name_label.pack(side=tk.LEFT, padx=2)
            name_label.bind("<Button-1>", lambda e, n=name: self._show_trajectory_info(n))
            
            # 删除按钮
            del_btn = ttk.Button(traj_frame, text="×", width=2, 
                               command=lambda n=name: self._remove_trajectory(n))
            del_btn.pack(side=tk.RIGHT)
        
        # 更新Canvas滚动区域
        self.traj_list_frame.update_idletasks()
        self.traj_canvas.configure(scrollregion=self.traj_canvas.bbox("all"))
    
    def _toggle_visibility(self, name, var):
        """切换轨迹可见性"""
        self.trajectories[name]['visible'] = var.get()
        self._update_plot()
    
    def _change_color(self, name):
        """更改轨迹颜色"""
        color = askcolor(color=self.trajectories[name]['color'], title="选择颜色")[1]
        if color:
            self.trajectories[name]['color'] = color
            self._update_trajectory_list()
            self._update_plot()
    
    def _remove_trajectory(self, name):
        """移除轨迹"""
        if name in self.trajectories:
            del self.trajectories[name]
            self._update_trajectory_list()
            self._update_plot()
            self._clear_info()
    
    def _clear_all(self):
        """清除所有轨迹"""
        self.trajectories.clear()
        self.color_index = 0
        self._update_trajectory_list()
        self._update_plot()
        self._clear_info()
    
    def _show_trajectory_info(self, name):
        """显示轨迹信息"""
        if name not in self.trajectories:
            return
        
        df = self.trajectories[name]['data']
        
        info = f"""轨迹名称: {name}
数据点数: {len(df)}

位置统计:
  X: {df['x'].min():.2f} ~ {df['x'].max():.2f} (mm)
  Y: {df['y'].min():.2f} ~ {df['y'].max():.2f} (mm)
  Z: {df['z'].min():.2f} ~ {df['z'].max():.2f} (mm)

起点: ({df['x'].iloc[0]:.2f}, {df['y'].iloc[0]:.2f}, {df['z'].iloc[0]:.2f})
终点: ({df['x'].iloc[-1]:.2f}, {df['y'].iloc[-1]:.2f}, {df['z'].iloc[-1]:.2f})
"""
        
        # 如果有姿态数据
        if 'rx' in df.columns and 'ry' in df.columns and 'rz' in df.columns:
            info += f"""
姿态统计 (度):
  RX: {df['rx'].min():.2f} ~ {df['rx'].max():.2f}
  RY: {df['ry'].min():.2f} ~ {df['ry'].max():.2f}
  RZ: {df['rz'].min():.2f} ~ {df['rz'].max():.2f}
"""
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)
    
    def _clear_info(self):
        """清除信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def _set_view(self, elev, azim, roll):
        """设置视角"""
        self.ax.view_init(elev=elev, azim=azim, roll=roll)
        self.canvas.draw()
    
    def _update_plot(self):
        """更新3D绘图"""
        self.ax.clear()
        
        has_visible = False
        
        for name, traj_data in self.trajectories.items():
            if not traj_data['visible']:
                continue
            
            has_visible = True
            df = traj_data['data']
            color = traj_data['color']
            
            # 绘制轨迹线
            self.ax.plot(df['x'], df['y'], df['z'], 
                        color=color, linewidth=2, label=name, alpha=0.8)
            
            # 显示数据点
            if self.show_points_var.get():
                self.ax.scatter(df['x'], df['y'], df['z'], 
                              c=color, s=20, alpha=0.6)
            
            # 标记起点和终点
            self.ax.scatter([df['x'].iloc[0]], [df['y'].iloc[0]], [df['z'].iloc[0]], 
                          c=color, s=100, marker='o', edgecolors='black', linewidths=2, 
                          label=f'{name} (起点)')
            self.ax.scatter([df['x'].iloc[-1]], [df['y'].iloc[-1]], [df['z'].iloc[-1]], 
                          c=color, s=100, marker='s', edgecolors='black', linewidths=2,
                          label=f'{name} (终点)')
            
            # 显示姿态箭头
            if self.show_orientation_var.get() and 'rx' in df.columns:
                self._draw_orientation_arrows(df, color)
        
        # 设置标签
        self.ax.set_xlabel('X (mm)', fontsize=10)
        self.ax.set_ylabel('Y (mm)', fontsize=10)
        self.ax.set_zlabel('Z (mm)', fontsize=10)
        
        if has_visible:
            self.ax.set_title('3D轨迹可视化', fontsize=12, fontweight='bold')
            # 添加图例
            self.ax.legend(loc='upper left', fontsize=8)
        else:
            self.ax.set_title('3D轨迹可视化 - 没有可见的轨迹', fontsize=12, fontweight='bold')
        
        # 设置网格
        self.ax.grid(self.show_grid_var.get())
        
        # 设置等比例
        if self.equal_aspect_var.get() and has_visible:
            self._set_equal_aspect()
        
        self.canvas.draw()
    
    def _draw_orientation_arrows(self, df, color, step=10):
        """绘制姿态箭头"""
        arrow_length = 20  # 箭头长度
        
        for i in range(0, len(df), step):
            x, y, z = df['x'].iloc[i], df['y'].iloc[i], df['z'].iloc[i]
            
            # 获取旋转角度（转换为弧度）
            rx = np.radians(df['rx'].iloc[i]) if 'rx' in df.columns else 0
            ry = np.radians(df['ry'].iloc[i]) if 'ry' in df.columns else 0
            rz = np.radians(df['rz'].iloc[i]) if 'rz' in df.columns else 0
            
            # 简化的旋转矩阵（仅用于可视化）
            # X轴箭头（红色）
            dx = arrow_length * np.cos(ry) * np.cos(rz)
            dy = arrow_length * np.cos(ry) * np.sin(rz)
            dz = arrow_length * np.sin(ry)
            self.ax.quiver(x, y, z, dx, dy, dz, color='red', arrow_length_ratio=0.3, alpha=0.5)
            
            # Y轴箭头（绿色）
            dx = arrow_length * (-np.cos(rx) * np.sin(rz) + np.sin(rx) * np.sin(ry) * np.cos(rz))
            dy = arrow_length * (np.cos(rx) * np.cos(rz) + np.sin(rx) * np.sin(ry) * np.sin(rz))
            dz = arrow_length * np.sin(rx) * np.cos(ry)
            self.ax.quiver(x, y, z, dx, dy, dz, color='green', arrow_length_ratio=0.3, alpha=0.5)
            
            # Z轴箭头（蓝色）
            dx = arrow_length * (np.sin(rx) * np.sin(rz) + np.cos(rx) * np.sin(ry) * np.cos(rz))
            dy = arrow_length * (-np.sin(rx) * np.cos(rz) + np.cos(rx) * np.sin(ry) * np.sin(rz))
            dz = arrow_length * np.cos(rx) * np.cos(ry)
            self.ax.quiver(x, y, z, dx, dy, dz, color='blue', arrow_length_ratio=0.3, alpha=0.5)
    
    def _set_equal_aspect(self):
        """设置等比例坐标轴"""
        all_x, all_y, all_z = [], [], []
        
        for traj_data in self.trajectories.values():
            if traj_data['visible']:
                df = traj_data['data']
                all_x.extend(df['x'].values)
                all_y.extend(df['y'].values)
                all_z.extend(df['z'].values)
        
        if all_x:
            max_range = max(
                max(all_x) - min(all_x),
                max(all_y) - min(all_y),
                max(all_z) - min(all_z)
            ) / 2.0
            
            mid_x = (max(all_x) + min(all_x)) / 2.0
            mid_y = (max(all_y) + min(all_y)) / 2.0
            mid_z = (max(all_z) + min(all_z)) / 2.0
            
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def _save_figure(self):
        """保存图片"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"图片已保存到: {file_path}")


def main():
    """主函数"""
    root = tk.Tk()
    app = Trajectory3DViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
