    def _vision_loop(self):
        while self.is_running:
            loop_start_time = time.perf_counter()
            
            # 1. 尝试获取当前追踪目标
            target_pos, target_id = None, None
            if hasattr(self.robot_view, 'renderer') and self.robot_view.renderer:
                target_pos, target_id = self.robot_view.renderer.get_tracking_target()
                
            if target_pos is not None and target_id is not None:
                # 获取真实 TCP 用于计算误差 (单位转换: mm -> m)
                tcp_mm = self.controller.state.get_tcp()
                tcp_m = [tcp_mm[0]/1000.0, tcp_mm[1]/1000.0, tcp_mm[2]/1000.0]
                
                # 计算当前末端相对于小球实际位置的水平误差 (用于判定是否截获)
                error_x = tcp_m[0] - target_pos[0]
                error_y = tcp_m[1] - target_pos[1]
                real_xy_distance = math.sqrt(error_x**2 + error_y**2)

                if self.use_sim_algorithm:
                    if self.last_target_id != target_id:
                        self.last_target_id = target_id
                        self.state = "HOVERING"
                        self.hover_count = 0
                        self.sim_virtual_pos = np.array(tcp_m)
                        self.sim_integral_error = np.zeros(3)
                        print(f"\n--- [仿真状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")
                    
                    if self.sim_virtual_pos is None:
                        self.sim_virtual_pos = np.array(tcp_m)
                    
                    # 基础物理预判 (补偿系统通信与执行延迟)
                    base_feedforward_time = (self.control_interval * self.look_ahead_frames) + 0.05
                    predicted_target_y = target_pos[1] + (self.conveyor_speed_y * base_feedforward_time)

                    # 确定当前的 3D 目标坐标 (针对 TCP)
                    desired_tcp_pos = np.array([target_pos[0], predicted_target_y, target_pos[2] + self.hover_height + self.tool_length])
                    
                    if self.state == "HOVERING":
                        # 漏斗式下压：根据 XY 误差动态调整高度，防止目标跑得快时视野过小而丢失
                        dynamic_hover_height = max(self.hover_height, min(0.15, real_xy_distance * 1.5))
                        desired_tcp_pos[2] = target_pos[2] + dynamic_hover_height + self.tool_length
                        
                        actual_tip_z_cur = tcp_m[2] - self.tool_length
                        target_final_hover_z = target_pos[2] + self.hover_height
                        if real_xy_distance < self.xy_threshold and abs(actual_tip_z_cur - target_final_hover_z) < 0.070:
                            self.hover_count += 1
                        else:
                            self.hover_count = 0
                            
                        if self.hover_count >= 5:
                            if getattr(self, 'hover_only_mode', False):
                                if self.hover_count % 60 == 0:
                                    print(f"--- [仿真状态机] 仅悬停模式：维持 HOVERING ---")
                            else:
                                print(f"--- [仿真状态机] 目标稳定截获，进入 APPROACHING ---")
                                self.state = "APPROACHING"
                                self.hover_count = 0
                                
                    elif self.state == "APPROACHING":
                        # 下压碰球
                        desired_tcp_pos[2] = self.target_z_surface - 0.010 + self.tool_length
                        actual_tip_z = tcp_m[2] - self.tool_length
                        if actual_tip_z <= 0.265:
                            print(f"--- [仿真状态机] 触碰小球，进入 RETURNING ---")
                            self.state = "RETURNING"
                            self.robot_view.renderer.mark_target_reached()
                            
                    elif self.state == "RETURNING":
                        # 抬起复位
                        desired_tcp_pos[2] = target_pos[2] + self.hover_height + self.tool_length
                        actual_tip_z = tcp_m[2] - self.tool_length
                        if actual_tip_z >= target_pos[2] + self.hover_height - 0.01:
                            print(f"--- [仿真状态机] 复位完成，回到 OBSERVING ---")
                            self.state = "OBSERVING"
                            self.last_target_id = None

                    # === 仿真算法控制律 (PI + 速度前馈 + 严格步长限幅) ===
                    dt = self.vision_interval
                    feedforward_vel = np.array([0.0, self.conveyor_speed_y, 0.0])
                    
                    # 使用虚拟位置进行误差计算和积分，避免物理机器人的延迟导致指令"被拽回"
                    error = desired_tcp_pos - self.sim_virtual_pos
                    self.sim_integral_error += error * dt
                    
                    norm_i = np.linalg.norm(self.sim_integral_error)
                    if norm_i > self.sim_integral_limit:
                        self.sim_integral_error = (self.sim_integral_error / norm_i) * self.sim_integral_limit
                        
                    v_cmd = (self.sim_kp * error) + (self.sim_ki * self.sim_integral_error) + feedforward_vel
                    delta_pos = v_cmd * dt
                    
                    step_distance = np.linalg.norm(delta_pos)
                    if step_distance > self.sim_max_step:
                        delta_pos = (delta_pos / step_distance) * self.sim_max_step
                        
                    self.sim_virtual_pos += delta_pos
                    final_target_pos = self.sim_virtual_pos.tolist()
                    self.current_z_target = final_target_pos[2] - self.tool_length # 同步用于打印

                elif self.use_dynamic_offset:
                    # ========================================================
                    # 自适应隐性 Offset 算法 (纯前馈 + 碰球后自适应校准)
                    # ========================================================
                    if self.last_target_id != target_id:
                        self.last_target_id = target_id
                        self.state = "HOVERING"
                        self.hover_count = 0
                        self.current_z_target = target_pos[2] + self.hover_height
                        print(f"\n--- [自适应Offset状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")
                        
                    # 计算基于自适应 Offset 的预测位置
                    total_latency_seconds = (self.control_interval * self.look_ahead_frames) + self.system_latency_offset
                    predicted_target_y = target_pos[1] + (self.conveyor_speed_y * total_latency_seconds)

                    if self.state == "HOVERING":
                        # 漏斗式下压
                        dynamic_hover_height = max(self.hover_height, min(0.15, real_xy_distance * 1.5))
                        self.current_z_target = target_pos[2] + dynamic_hover_height
                        
                        actual_tip_z_cur = tcp_m[2] - self.tool_length
                        target_final_hover_z = target_pos[2] + self.hover_height
                        if real_xy_distance < self.xy_threshold and abs(actual_tip_z_cur - target_final_hover_z) < 0.070:
                            self.hover_count += 1
                        else:
                            self.hover_count = 0
                            
                        if self.hover_count >= 5:
                            if getattr(self, 'hover_only_mode', False):
                                if self.hover_count % 60 == 0:
                                    print(f"--- [自适应Offset状态机] 仅悬停模式：维持 HOVERING ---")
                            else:
                                print(f"--- [自适应Offset状态机] 目标已稳定截获，进入 APPROACHING ---")
                                self.state = "APPROACHING"
                                self.hover_count = 0
                                if self.use_blind_tracking:
                                    self.locked_x = target_pos[0]
                                    self.locked_y = predicted_target_y
                                    print(f"--- [盲抓模式] 已锁定坐标 X:{self.locked_x:.3f}, Y:{self.locked_y:.3f} ---")

                    elif self.state == "APPROACHING":
                        self.locked_y += self.conveyor_speed_y * self.vision_interval
                        
                        approach_time_budget = 0.5 # 秒
                        dynamic_approach_speed_z = self.hover_height / approach_time_budget
                        
                        self.current_z_target -= dynamic_approach_speed_z * self.vision_interval
                        if self.current_z_target < self.target_z_surface - 0.010:
                            self.current_z_target = self.target_z_surface - 0.010
                        
                        actual_tip_z = tcp_m[2] - self.tool_length
                        
                        if actual_tip_z <= 0.265:
                            final_err_x = tcp_m[0] - target_pos[0]
                            final_err_y = tcp_m[1] - target_pos[1]
                            final_dist_mm = math.sqrt(final_err_x**2 + final_err_y**2) * 1000.0
                            
                            print(f"--- [自适应Offset状态机] 已触碰小球表面，最终物理误差: {final_dist_mm:.2f}mm (X:{final_err_x*1000:.1f}, Y:{final_err_y*1000:.1f}) ---")
                            
                            # [自适应校准 - Auto Calibration]
                            # err_y = (真实物理延迟 - 我们的预判延迟) * 速度
                            # 如果 err_y > 0 (超前)，说明我们的预判延迟给大了，需要减小 offset
                            if abs(self.conveyor_speed_y) > 0.001:
                                learning_rate = 0.3  # 学习率
                                time_correction = final_err_y / self.conveyor_speed_y
                                self.system_latency_offset -= time_correction * learning_rate
                                # 限制范围，避免异常参数 (0ms ~ 300ms)
                                self.system_latency_offset = max(0.0, min(0.3, self.system_latency_offset))
                                print(f"--- [自适应校准] 自动修正延迟系数 -> 新 offset: {self.system_latency_offset:.4f}s ---")

                            self.state = "RETURNING"
                            self.robot_view.renderer.mark_target_reached()

                    elif self.state == "RETURNING":
                        self.locked_y += self.conveyor_speed_y * self.vision_interval
                        self.current_z_target += (self.approach_speed_z * 2) * self.vision_interval
                        if self.current_z_target >= target_pos[2] + self.hover_height:
                            self.current_z_target = target_pos[2] + self.hover_height
                            print(f"--- [自适应Offset状态机] 复位完成，回到 OBSERVING 状态 ---")
                            self.state = "OBSERVING"
                            self.last_target_id = None
                    
                    if self.state in ["APPROACHING", "RETURNING"]:
                        final_target_pos = [self.locked_x, self.locked_y, self.current_z_target + self.tool_length]
                    else:
                        final_target_pos = [target_pos[0], predicted_target_y, self.current_z_target + self.tool_length]

                else:
                    # 如果是新进来的目标，立刻重置为 HOVERING
                    if self.last_target_id != target_id:
                        self.last_target_id = target_id
                        self.state = "HOVERING"
                        self.hover_count = 0  # 重置计数
                        self.current_z_target = target_pos[2] + self.hover_height
                        print(f"\n--- [状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")

                    # ==========================
                    # 前馈速度补偿与截击 (Feedforward & Intercept)
                    # ==========================
                    # 基础物理预判 (固定的前馈时间，大约为预瞄加通信时间)
                    base_feedforward_time = (self.control_interval * self.look_ahead_frames) + 0.05
                    base_predicted_y = target_pos[1] + (self.conveyor_speed_y * base_feedforward_time)
                    
                    # ==========================
                    # PI 闭环补偿控制 (仅在 Y 轴方向)
                    # ==========================
                    # 目标：让机械臂在 Y 轴跟上小球，即 error_y -> 0
                    # 如果小球比机械臂更靠前(沿着Y轴正方向跑)，error_y 为负，我们需要把预测点再往前推一点。
                    # 所以我们期望机械臂的 Y > 现在的 TCP Y，即补偿量为负的 error_y 相关项。
                    
                    if self.state in ["HOVERING", "APPROACHING"]:
                        # 计算 PID 补偿
                        self.pi_integral_y += error_y * self.vision_interval
                        
                        # 微分项计算 (当前误差 - 上次误差) / 时间
                        derivative_y = (error_y - self.pi_last_error_y) / self.vision_interval
                        self.pi_last_error_y = error_y
                        
                        # 限制积分饱和 (防积分风饱和)
                        max_integral = 0.2  # 最大积分为20cm
                        self.pi_integral_y = max(-max_integral, min(max_integral, self.pi_integral_y))
                        
                        # 计算控制输出: P * 误差 + I * 积分 + D * 微分
                        # 如果小球在前面 (target_y > tcp_y)，error_y 为负。我们需要增加 predicted_y，所以减去控制输出。
                        self.pi_compensation_y = - (self.pi_kp * error_y + self.pi_ki * self.pi_integral_y + self.pi_kd * derivative_y)
                    else:
                        # 如果丢失或在复位，清空积分器和微分器
                        self.pi_integral_y = 0.0
                        self.pi_last_error_y = 0.0
                        self.pi_compensation_y = 0.0
                        
                    # 最终预测目标 = 基础前馈位置 + PI 动态补偿
                    predicted_target_y = base_predicted_y + self.pi_compensation_y

                    # ==========================
                    # 状态机核心控制流
                    # ==========================
                    if self.state == "HOVERING":
                        # 【优化】漏斗式下压：根据 XY 误差动态调整高度，防止目标跑得快时视野过小而丢失
                        # 当 XY 误差较大时，保持较高的高度 (最大可额外抬高 15cm)，随着对准逐渐下降
                        dynamic_hover_height = max(self.hover_height, min(0.15, real_xy_distance * 1.5))
                        self.current_z_target = target_pos[2] + dynamic_hover_height
                        
                        actual_tip_z_cur = tcp_m[2] - self.tool_length
                        
                        # 判定条件：水平对准且高度已降到真实的悬停位附近
                        target_final_hover_z = target_pos[2] + self.hover_height
                        if real_xy_distance < self.xy_threshold and abs(actual_tip_z_cur - target_final_hover_z) < 0.070:
                            self.hover_count += 1
                        else:
                            self.hover_count = 0
                        
                        if self.hover_count >= 5:
                            if getattr(self, 'hover_only_mode', False):
                                if self.hover_count % 60 == 0: # 避免刷屏，每秒打印一次
                                    print(f"--- [状态机] 仅悬停模式：维持 HOVERING，不进入 APPROACHING (计数={self.hover_count}) ---")
                            else:
                                print(f"--- [状态机] 目标已稳定截获 (计数={self.hover_count})，进入 APPROACHING ---")
                                self.state = "APPROACHING"
                                self.hover_count = 0
                                if self.use_blind_tracking:
                                    self.locked_x = target_pos[0]
                                    self.locked_y = predicted_target_y
                                    print(f"--- [盲抓模式] 已锁定坐标 X:{self.locked_x:.3f}, Y:{self.locked_y:.3f} ---")

                    elif self.state == "APPROACHING":
                        self.locked_y += self.conveyor_speed_y * self.vision_interval
                        
                        approach_time_budget = 0.5 # 秒
                        dynamic_approach_speed_z = self.hover_height / approach_time_budget
                        
                        self.current_z_target -= dynamic_approach_speed_z * self.vision_interval
                        if self.current_z_target < self.target_z_surface - 0.010:
                            self.current_z_target = self.target_z_surface - 0.010
                        
                        actual_tip_z = tcp_m[2] - self.tool_length
                        
                        if actual_tip_z <= 0.265:
                            final_err_x = tcp_m[0] - target_pos[0]
                            final_err_y = tcp_m[1] - target_pos[1]
                            final_dist_mm = math.sqrt(final_err_x**2 + final_err_y**2) * 1000.0
                            
                            print(f"--- [状态机] 已触碰小球表面，最终物理误差: {final_dist_mm:.2f}mm (X:{final_err_x*1000:.1f}, Y:{final_err_y*1000:.1f}) ---")
                            
                            self.state = "RETURNING"
                            self.robot_view.renderer.mark_target_reached()

                    elif self.state == "RETURNING":
                        self.locked_y += self.conveyor_speed_y * self.vision_interval
                        self.current_z_target += (self.approach_speed_z * 2) * self.vision_interval
                        if self.current_z_target >= target_pos[2] + self.hover_height:
                            self.current_z_target = target_pos[2] + self.hover_height
                            print(f"--- [状态机] 复位完成，回到 OBSERVING 状态 ---")
                            self.state = "OBSERVING"
                            self.last_target_id = None
                    
                    if self.state in ["APPROACHING", "RETURNING"]:
                        final_target_pos = [self.locked_x, self.locked_y, self.current_z_target + self.tool_length]
                    else:
                        final_target_pos = [target_pos[0], predicted_target_y, self.current_z_target + self.tool_length]
                
                # 写入缓冲区供控制线程读取
                with self._target_lock:
                    self._target_buffer['pos'] = final_target_pos
                    self._target_buffer['timestamp'] = time.perf_counter()
                    self._target_buffer['valid'] = True
                    self._target_buffer['state'] = self.state
                    self._target_buffer['has_target'] = True
                    self._target_buffer['conveyor_speed'] = self.conveyor_speed_y
                    
                if self.state in ["HOVERING", "APPROACHING"] and int(loop_start_time * 10) % 10 == 0: 
                    actual_tip_z = tcp_m[2] - self.tool_length
                    print(f"[{self.state}] 目标Z:{self.current_z_target:.3f}m | 实际尖端Z:{actual_tip_z:.3f}m | XY误差:{real_xy_distance*1000.0:.1f}mm")

            else:
                if self.state != "OBSERVING":
                    print(f"--- [状态机] 目标丢失，返回初始位置 ---")
                    self.state = "OBSERVING"
                    self.last_target_id = None
                
                with self._target_lock:
                    self._target_buffer['valid'] = False
                    self._target_buffer['state'] = "OBSERVING"
                    self._target_buffer['has_target'] = False

            # 频率控制 (60Hz)
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = max(0, self.vision_interval - elapsed)
            time.sleep(sleep_time)

