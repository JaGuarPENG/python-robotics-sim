import websocket
import struct
import json
import time
import hashlib
import statistics
import matplotlib.pyplot as plt

class RobotLatencyTester:
    def __init__(self, ip, port=5888):
        self.uri = f"ws://{ip}:{port}"
        self.ws = None
        self.is_connected = False

    def connect(self):
        try:
            # 禁用 Nagle 算法 (TCP_NODELAY) 通常由 websocket 库底层处理或操作系统决定
            # 这里我们主要关注应用层建立连接
            self.ws = websocket.WebSocket()
            self.ws.connect(self.uri, timeout=3)
            self.is_connected = True
            print(f"[连接] 成功连接到 {self.uri}")
            return True
        except Exception as e:
            print(f"[连接] 失败: {e}")
            return False

    def close(self):
        if self.ws:
            self.ws.close()

    def _pack_header(self, msg_len: int) -> bytes:
        return struct.pack('<IIQqqq', msg_len, 0x01, 0x1000, 0xA1B2C3D4, 0, 0)

    def send_and_measure(self, cmd_str: str):
        """发送指令并测量往返耗时 (秒)"""
        if not self.is_connected: return None

        try:
            # 准备数据
            payload = cmd_str.encode('utf-8')
            header = self._pack_header(len(payload))
            full_packet = header + payload

            # --- 计时开始 ---
            start_time = time.perf_counter()
            
            # 发送
            self.ws.send(full_packet, opcode=websocket.ABNF.OPCODE_BINARY)
            
            # 接收 (阻塞等待直到收到数据)
            resp = self.ws.recv()
            
            # --- 计时结束 ---
            end_time = time.perf_counter()
            
            return end_time - start_time

        except Exception as e:
            print(f"[测试] 通信异常: {e}")
            return None

    def login(self, user="Engineer", pwd=""):
        # 登录只需做一次，不需要测量延迟
        pwd_md5 = hashlib.md5(pwd.encode('utf-8')).hexdigest()
        cmd = f"login --user={user} --pwd={pwd_md5}"
        self.send_and_measure(cmd)
        # 简单延时确保登录状态生效
        time.sleep(0.5)

    def run_benchmark(self, count=100, interval=0.05):
        """
        运行基准测试
        :param count: 测试次数
        :param interval: 每次测试间隔(秒)
        """
        if not self.connect(): return

        print("正在登录...")
        self.login("Engineer", "000000")

        latencies = []
        print(f"\n开始延迟测试 (发送 'get' 指令, 共 {count} 次)...")
        print("-" * 40)

        try:
            for i in range(count):
                # 使用 'get' 指令，因为它数据量适中且无副作用
                rtt = self.send_and_measure("get")
                
                if rtt is not None:
                    # 转换为毫秒
                    rtt_ms = rtt * 1000
                    latencies.append(rtt_ms)
                    # 实时打印前几个和每10个的数据
                    if i < 5 or i % 10 == 0:
                        print(f"Seq {i+1}: {rtt_ms:.2f} ms")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        finally:
            self.close()

        # --- 结果分析 ---
        if latencies:
            avg_lat = statistics.mean(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0

            print("-" * 40)
            print(f"测试结果统计 ({len(latencies)} 次成功):")
            print(f"平均延迟 (Avg): {avg_lat:.2f} ms")
            print(f"最小延迟 (Min): {min_lat:.2f} ms")
            print(f"最大延迟 (Max): {max_lat:.2f} ms")
            print(f"抖动 (Jitter/StdDev): {jitter:.2f} ms")
            print("-" * 40)
            
            # 简单建议
            if avg_lat < 10:
                print("✅ 状态: 极佳 (适合高频控制)")
            elif avg_lat < 30:
                print("✅ 状态: 良好 (适合一般控制)")
            else:
                print("⚠️ 状态: 较高 (可能影响实时闭环控制)")

            # --- 绘图 ---
            self.plot_results(latencies)
        else:
            print("没有收集到有效数据")

    def plot_results(self, latencies):
        plt.figure(figsize=(10, 6))
        
        # 1. 时序图
        plt.subplot(2, 1, 1)
        plt.plot(latencies, 'b.-', alpha=0.7)
        plt.axhline(y=statistics.mean(latencies), color='r', linestyle='--', label='Average')
        plt.title('Communication Latency Test (RTT)')
        plt.ylabel('Latency (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 直方图
        plt.subplot(2, 1, 2)
        plt.hist(latencies, bins=20, color='green', alpha=0.7)
        plt.title('Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 请修改IP
    tester = RobotLatencyTester("192.168.1.5", 5888)
    
    # 运行测试：测试 100 次，每次间隔 0.02秒 (50Hz频率发送)
    tester.run_benchmark(count=100, interval=0.02)