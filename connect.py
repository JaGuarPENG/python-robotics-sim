import websocket
import struct
import json
import time
import hashlib

class ArisRobotClient:
    def __init__(self, ip, port=5888, timeout=10):
        self.uri = f"ws://{ip}:{port}"
        self.ws = None
        self.is_connected = False
        self.timeout = timeout  # 保存超时时间设置

    def connect(self):
        try:
            self.ws = websocket.WebSocket()
            # 【修改点1】这里使用类初始化时传入的 timeout，建议设大一点
            self.ws.connect(self.uri, timeout=self.timeout) 
            self.is_connected = True
            print(f"[连接] 成功: {self.uri}")
            return True
        except Exception as e:
            print(f"[连接] 失败: {e}")
            return False

    def close(self):
        if self.ws: self.ws.close()

    def _pack_header(self, msg_len):
        return struct.pack('<IIQqqq', msg_len, 0x01, 0x1000, 0xA1B2C3D4, 0, 0)

    def _send_raw_command(self, cmd_str, need_reply=True):
        """
        need_reply: 如果有些指令确实不回包，可以设为 False 避免死等
        """
        if not self.is_connected: return None
        try:
            payload = cmd_str.encode('utf-8')
            full_packet = self._pack_header(len(payload)) + payload
            self.ws.send(full_packet, opcode=websocket.ABNF.OPCODE_BINARY)
            print(f"[发送] {cmd_str}")
            
            if not need_reply:
                return None

            # 【修改点2】接收时增加异常处理，防止超时导致程序崩溃
            resp = self.ws.recv()
            return resp[40:] if len(resp) >= 40 else None
            
        except websocket.WebSocketTimeoutException:
            print(f"[超时] 指令发送成功，但在 {self.timeout}秒内未收到回复 (可能动作时间过长)")
            return None
        except Exception as e:
            print(f"[异常] {e}")
            return None
    
    def _parse_json(self, raw_bytes):
        """辅助函数：解析JSON回复"""
        if not raw_bytes: return None
        try:
            data = json.loads(raw_bytes.decode('utf-8'))
            if 'ret_context' in data and isinstance(data['ret_context'], str):
                try:
                    data['ret_context'] = json.loads(data['ret_context'])
                except: pass
            return data
        except: return None

    # --- 业务指令 ---

    def login(self, user="Engineer", pwd=""):
        pwd_md5 = hashlib.md5(pwd.encode('utf-8')).hexdigest()
        self._send_raw_command(f"login --user={user} --pwd={pwd_md5}")

    def manual_enable(self):
        self._send_raw_command("manual_en")

    def movej(self, joints):
        """发送关节运动指令"""
        try:
            if len(joints) != 6:
                print(f"[错误] movej 需要 6 个关节角度")
                return None
            
            joint_strs = [f"DOUBLE{{{j:.6f}}}" for j in joints]
            inner_str = ",".join(joint_strs)
            # 注意：这里使用了你确认过的正确格式，如果是 SixAxisModel 请自行替换
            cmd = f"manual_mvaj --pos=JointTarget{{UrModel_JointTarget{{{inner_str}}}}}"
            
            # 发送指令，这里可能会阻塞直到机器人回复
            response = self._send_raw_command(cmd)
            return response
            
        except Exception as e:
            print(f"[MoveJ] 执行出错: {e}")
            return None

    def get_status(self):
        """获取机器人状态"""
        resp = self._send_raw_command("get")
        return self._parse_json(resp)

    # 【修改点3】新增：智能等待函数 TODO: 优化等待逻辑, 目前咋样状态都是normal，判断不了
    def wait_for_stop(self, interval=0.5):
        """
        轮询机器人状态，直到机器人停止运动 (status != 'Running')
        """
        print("[等待] 正在等待动作完成...", end="", flush=True)
        while True:
            status_data = self.get_status()
            if status_data and 'ret_context' in status_data:
                # 获取机器人状态字符串
                # 注意：不同版本 key 可能不同，可能是 'robot_msg' -> 'status'
                # 或者是直接 'status'，请根据 get 指令的实际打印结果调整
                try:
                    status = status_data['ret_context']['robot_msg']['status']
                except KeyError:
                    # 如果找不到状态字段，默认跳过
                    print(" (无法获取状态字段，跳过等待) ")
                    break

                # 常见的状态有: Ready, Running, Error, Paused
                if status != "Running":
                    print(f" [完成] 当前状态: {status}")
                    break
            
            print(".", end="", flush=True)
            time.sleep(interval)

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 【建议】超时时间设为 10秒 或 20秒，取决于你的动作幅度
    robot = ArisRobotClient("192.168.1.5", 5888, timeout=20)
    
    if robot.connect():
        # 1. 登录 & 上电
        robot.login("Engineer", "000000")
        time.sleep(0.5)
        robot.manual_enable()
        time.sleep(1)
        
        # 2. 定义目标
        target_pos_1 = [0, 0, 0, 0, 0, 0]
        
        # 3. 发送运动指令
        print("\n--- 移动到点 1 ---")
        robot.movej(target_pos_1)
        
        # 【改进】使用轮询等待替代死板的 sleep
        # 这样既不会超时（因为get指令很快），又能准确知道何时结束
        robot.wait_for_stop() 
        
        print("\n--- 动作已确认完成，程序结束 ---")
        robot.close()