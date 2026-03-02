import math
import numpy as np
from .base import BaseTrackingAlgorithm

class SimTrackingAlgorithm(BaseTrackingAlgorithm):
    def reset(self):
        self.ctx.sim_integral_error = np.zeros(3)
        self.ctx.sim_virtual_pos = None

    def execute_vision_cycle(self, target_pos, target_id, tcp_m, loop_start_time):
        ctx = self.ctx
        
        error_x = tcp_m[0] - target_pos[0]
        error_y = tcp_m[1] - target_pos[1]
        real_xy_distance = math.sqrt(error_x**2 + error_y**2)

        if ctx.last_target_id != target_id:
            ctx.last_target_id = target_id
            ctx.state = "HOVERING"
            ctx.hover_count = 0
            ctx.sim_virtual_pos = np.array(tcp_m)
            ctx.sim_integral_error = np.zeros(3)
            print(f"--- [仿真状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")
        
        if ctx.sim_virtual_pos is None:
            ctx.sim_virtual_pos = np.array(tcp_m)
        
        base_feedforward_time = (ctx.control_interval * ctx.look_ahead_frames) + 0.05
        predicted_target_y = target_pos[1] + (ctx.conveyor_speed_y * base_feedforward_time)

        desired_tcp_pos = np.array([target_pos[0], predicted_target_y, target_pos[2] + ctx.hover_height + ctx.tool_length])
        
        if ctx.state == "HOVERING":
            dynamic_hover_height = max(ctx.hover_height, min(0.15, real_xy_distance * 1.5))
            desired_tcp_pos[2] = target_pos[2] + dynamic_hover_height + ctx.tool_length
            
            actual_tip_z_cur = tcp_m[2] - ctx.tool_length
            target_final_hover_z = target_pos[2] + ctx.hover_height
            if real_xy_distance < ctx.xy_threshold and abs(actual_tip_z_cur - target_final_hover_z) < 0.070:
                ctx.hover_count += 1
            else:
                ctx.hover_count = 0
                
            if ctx.hover_count >= 5:
                if getattr(ctx, 'hover_only_mode', False):
                    if ctx.hover_count % 60 == 0:
                        print(f"--- [仿真状态机] 仅悬停模式：维持 HOVERING ---")
                else:
                    print(f"--- [仿真状态机] 目标稳定截获，进入 APPROACHING ---")
                    ctx.state = "APPROACHING"
                    ctx.hover_count = 0
                    
        elif ctx.state == "APPROACHING":
            desired_tcp_pos[2] = ctx.target_z_surface - 0.010 + ctx.tool_length
            actual_tip_z = tcp_m[2] - ctx.tool_length
            if actual_tip_z <= 0.265:
                print(f"--- [仿真状态机] 触碰小球，进入 RETURNING ---")
                ctx.state = "RETURNING"
                ctx.robot_view.renderer.mark_target_reached()
                
        elif ctx.state == "RETURNING":
            desired_tcp_pos[2] = target_pos[2] + ctx.hover_height + ctx.tool_length
            actual_tip_z = tcp_m[2] - ctx.tool_length
            if actual_tip_z >= target_pos[2] + ctx.hover_height - 0.01:
                print(f"--- [仿真状态机] 复位完成，回到 OBSERVING ---")
                ctx.state = "OBSERVING"
                ctx.last_target_id = None

        dt = ctx.vision_interval
        feedforward_vel = np.array([0.0, ctx.conveyor_speed_y, 0.0])
        
        error = desired_tcp_pos - ctx.sim_virtual_pos
        ctx.sim_integral_error += error * dt
        
        norm_i = np.linalg.norm(ctx.sim_integral_error)
        if norm_i > ctx.sim_integral_limit:
            ctx.sim_integral_error = (ctx.sim_integral_error / norm_i) * ctx.sim_integral_limit
            
        v_cmd = (ctx.sim_kp * error) + (ctx.sim_ki * ctx.sim_integral_error) + feedforward_vel
        delta_pos = v_cmd * dt
        
        step_distance = np.linalg.norm(delta_pos)
        if step_distance > ctx.sim_max_step:
            delta_pos = (delta_pos / step_distance) * ctx.sim_max_step
            
        ctx.sim_virtual_pos += delta_pos
        final_target_pos = ctx.sim_virtual_pos.tolist()
        ctx.current_z_target = final_target_pos[2] - ctx.tool_length # 同步用于打印

        if ctx.state in ["HOVERING", "APPROACHING"] and int(loop_start_time * 10) % 10 == 0: 
            actual_tip_z = tcp_m[2] - ctx.tool_length
            print(f"[{ctx.state}] 目标Z:{ctx.current_z_target:.3f}m | 实际尖端Z:{actual_tip_z:.3f}m | XY误差:{real_xy_distance*1000.0:.1f}mm")

        return final_target_pos
