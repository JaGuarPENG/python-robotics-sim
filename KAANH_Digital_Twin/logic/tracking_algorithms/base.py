class BaseTrackingAlgorithm:
    def __init__(self, ctx):
        self.ctx = ctx  # 引用 ConveyorTrackingService 实例

    def reset(self):
        """算法复位（追踪开始或目标丢失时调用）"""
        pass

    def execute_vision_cycle(self, target_pos, target_id, tcp_m, loop_start_time):
        """
        视觉循环核心逻辑
        Returns:
            final_target_pos: list [x, y, z] 或 None（如果不需要更新）
        """
        raise NotImplementedError
