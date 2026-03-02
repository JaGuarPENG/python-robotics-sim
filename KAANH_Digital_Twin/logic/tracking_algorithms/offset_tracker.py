import math
from .base import BaseTrackingAlgorithm

class OffsetTrackingAlgorithm(BaseTrackingAlgorithm):
    def reset(self):
        pass

    def execute_vision_cycle(self, target_pos, target_id, tcp_m, loop_start_time):
        ctx = self.ctx
        
        error_x = tcp_m[0] - target_pos[0]
        error_y = tcp_m[1] - target_pos[1]
        real_xy_distance = math.sqrt(error_x**2 + error_y**2)

        if ctx.last_target_id != target_id:
            ctx.last_target_id = target_id
            ctx.state = "HOVERING"
            ctx.hover_count = 0
            ctx.current_z_target = target_pos[2] + ctx.hover_height
            print(f"--- [自适应Offset状态机] 发现新目标 ID={target_id}，进入 HOVERING (悬停同步) ---")
            
        total_latency_seconds = (ctx.control_interval * ctx.look_ahead_frames) + ctx.system_latency_offset
        predicted_target_y = target_pos[1] + (ctx.conveyor_speed_y * total_latency_seconds)

        if ctx.state == "HOVERING":
            dynamic_hover_height = max(ctx.hover_height, min(0.15, real_xy_distance * 1.5))
            ctx.current_z_target = target_pos[2] + dynamic_hover_height
            
            actual_tip_z_cur = tcp_m[2] - ctx.tool_length
            target_final_hover_z = target_pos[2] + ctx.hover_height
            if real_xy_distance < ctx.xy_threshold and abs(actual_tip_z_cur - target_final_hover_z) < 0.070:
                ctx.hover_count += 1
            else:
                ctx.hover_count = 0
                
            if ctx.hover_count >= 5:
                if getattr(ctx, 'hover_only_mode', False):
                    if ctx.hover_count % 60 == 0:
                        print(f"--- [自适应Offset状态机] 仅悬停模式：维持 HOVERING ---")
                else:
                    print(f"--- [自适应Offset状态机] 目标已稳定截获，进入 APPROACHING ---")
                    ctx.state = "APPROACHING"
                    ctx.hover_count = 0
                    if ctx.use_blind_tracking:
                        ctx.locked_x = target_pos[0]
                        ctx.locked_y = predicted_target_y
                        print(f"--- [盲抓模式] 已锁定坐标 X:{ctx.locked_x:.3f}, Y:{ctx.locked_y:.3f} ---")

        elif ctx.state == "APPROACHING":
            ctx.locked_y += ctx.conveyor_speed_y * ctx.vision_interval
            
            approach_time_budget = 0.5 # 秒
            dynamic_approach_speed_z = ctx.hover_height / approach_time_budget
            
            ctx.current_z_target -= dynamic_approach_speed_z * ctx.vision_interval
            if ctx.current_z_target < ctx.target_z_surface - 0.010:
                ctx.current_z_target = ctx.target_z_surface - 0.010
            
            actual_tip_z = tcp_m[2] - ctx.tool_length
            
            if actual_tip_z <= 0.265:
                final_err_x = tcp_m[0] - target_pos[0]
                final_err_y = tcp_m[1] - target_pos[1]
                final_dist_mm = math.sqrt(final_err_x**2 + final_err_y**2) * 1000.0
                
                print(f"--- [自适应Offset状态机] 已触碰小球表面，最终物理误差: {final_dist_mm:.2f}mm (X:{final_err_x*1000:.1f}, Y:{final_err_y*1000:.1f}) ---")
                
                if abs(ctx.conveyor_speed_y) > 0.001:
                    learning_rate = 0.3
                    time_correction = final_err_y / ctx.conveyor_speed_y
                    ctx.system_latency_offset -= time_correction * learning_rate
                    ctx.system_latency_offset = max(0.0, min(0.3, ctx.system_latency_offset))
                    print(f"--- [自适应校准] 自动修正延迟系数 -> 新 offset: {ctx.system_latency_offset:.4f}s ---")

                ctx.state = "RETURNING"
                ctx.robot_view.renderer.mark_target_reached()

        elif ctx.state == "RETURNING":
            ctx.locked_y += ctx.conveyor_speed_y * ctx.vision_interval
            ctx.current_z_target += (ctx.approach_speed_z * 2) * ctx.vision_interval
            if ctx.current_z_target >= target_pos[2] + ctx.hover_height:
                ctx.current_z_target = target_pos[2] + ctx.hover_height
                print(f"--- [自适应Offset状态机] 复位完成，回到 OBSERVING 状态 ---")
                ctx.state = "OBSERVING"
                ctx.last_target_id = None
        
        if ctx.state in ["APPROACHING", "RETURNING"]:
            final_target_pos = [ctx.locked_x, ctx.locked_y, ctx.current_z_target + ctx.tool_length]
        else:
            final_target_pos = [target_pos[0], predicted_target_y, ctx.current_z_target + ctx.tool_length]

        if ctx.state in ["HOVERING", "APPROACHING"] and int(loop_start_time * 10) % 10 == 0: 
            actual_tip_z = tcp_m[2] - ctx.tool_length
            print(f"[{ctx.state}] 目标Z:{ctx.current_z_target:.3f}m | 实际尖端Z:{actual_tip_z:.3f}m | XY误差:{real_xy_distance*1000.0:.1f}mm")

        return final_target_pos
