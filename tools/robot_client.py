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
            # 建立连接
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
        if not self.is_connected: return None
        try:
            payload = cmd_str.encode('utf-8')
            full_packet = self._pack_header(len(payload)) + payload
            self.ws.send(full_packet, opcode=websocket.ABNF.OPCODE_BINARY)
            
            if not need_reply:
                return None

            # 阻塞等待接收
            resp = self.ws.recv()
            return resp[40:] if len(resp) >= 40 else None
            
        except websocket.WebSocketTimeoutException:
            print(f"[超时] 指令发送成功，但在 {self.timeout}秒内未收到回复 (可能动作时间过长)")
            return None
        except Exception as e:
            print(f"[异常] {e}")
            return None
    
    def _parse_json(self, raw_bytes):
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

    def movej(self, joints, vels):
        """发送关节运动指令"""
        try:
            if len(joints) != 6:
                print(f"[错误] movej 需要 6 个关节角度")
                return None
            
            joint_strs = [f"DOUBLE{{{j:.6f}}}" for j in joints]
            joint_inner_str = ",".join(joint_strs)

            vel_strs = [f"DOUBLE{{{v:.6f}}}" for v in vels]
            vel_inner_str = ",".join(vel_strs)
  
            # 【修改】注意 Speed 后面加了三层花括号 {{{...}}} 
            # f-string中 {{ 转义为 {，所以 {{{ }}} 解析为 {内容}
            cmd = f"manual_mvaj --pos=JointTarget{{UrModel_JointTarget{{{joint_inner_str}}}}} --vel=Speed{{{vel_inner_str}}}"
            
            # 这里的 response 会一直阻塞直到机器人动作完成并返回
            response = self._send_raw_command(cmd)
            return response
            
        except Exception as e:
            print(f"[MoveJ] 执行出错: {e}")
            return None
        
    def set_jog_vel(self, percent):
        """设置JOG速度百分比 (0-100)"""
        percent = max(0, min(100, percent))
        self._send_raw_command(f"set_jog_vel --vel_percent={percent}")

    def set_pgm_vel(self, percent):
        """设置程序速度百分比 (0-100)"""
        percent = max(0, min(100, percent))
        self._send_raw_command(f"set_pgm_vel --vel_percent={percent}")

    def get_status(self):
        resp = self._send_raw_command("get")
        return self._parse_json(resp)
    
if __name__ == "__main__":
    client = ArisRobotClient("192.168.1.5", 5888)
    if client.connect():
        client.login("Engineer", "000000")
        time.sleep(0.5)
        client.manual_enable()
        print("已登录并使能机器人")

        client.set_jog_vel(100)
        client.set_pgm_vel(100)
        print("速度已设置为 100%")

        print("发送 MoveJ 指令...")
        joints = [0, -90, 90, 0, 90, 0]  # 目标关节角度
        vels = [50, 100, 50]            # 关节速度
        response = client.movej(joints, vels)
        print("MoveJ 指令完成，响应:")
        print(response)

        client.close()