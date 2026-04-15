"""
交互式机器人命令工具

用法:
    python tools/robot_cmd.py --ip 192.168.1.10

然后输入命令，如:
    logout
    login
    cl
    manual_en
    start_follower
    follower_cart
    get
    quit (退出)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.follower_client import FollowerClientWebSocket
import hashlib
import json

# 配置
ROBOT_IP = "192.168.1.10"
WS_PORT = 5999


def main():
    print("=" * 60)
    print("交互式机器人命令工具")
    print("=" * 60)
    print(f"连接: ws://{ROBOT_IP}:{WS_PORT}")
    print()
    print("特殊命令:")
    print("  login    - 自动登录 (Engineer/000000)")
    print("  get      - 获取状态")
    print("  status   - 显示详细状态")
    print("  quit     - 退出")
    print()

    client = FollowerClientWebSocket(ROBOT_IP, WS_PORT, timeout=15.0)

    if not client.connect():
        print("连接失败!")
        return

    print("连接成功! 输入命令 (quit 退出):\n")

    while True:
        try:
            cmd = input("> ").strip()

            if not cmd:
                continue

            if cmd.lower() == 'quit' or cmd.lower() == 'exit':
                break

            if cmd.lower() == 'login':
                # 自动登录
                pwd_md5 = hashlib.md5("000000".encode()).hexdigest()
                client.send_command(f"login --user=Engineer --pwd={pwd_md5}")

            elif cmd.lower() == 'get':
                # 获取状态
                client.send_get_status()
                if client.last_status:
                    print(f"  ret_code: {client.last_status.get('ret_code')}")

            elif cmd.lower() == 'status':
                # 显示详细状态
                client.send_get_status()
                if client.last_status:
                    try:
                        ret_context = client.last_status.get('ret_context', {})
                        if isinstance(ret_context, str):
                            ret_context = json.loads(ret_context)
                        robot_msg = ret_context.get('robot_msg', {})
                        print(f"  activate: {robot_msg.get('activate')}")
                        print(f"  status: {robot_msg.get('status')}")
                        print(f"  motion: {robot_msg.get('motion')}")
                        print(f"  op_mode: {robot_msg.get('op_mode')}")
                    except Exception as e:
                        print(f"  解析失败: {e}")

            else:
                # 发送原始命令
                client.send_command(cmd, timeout=15.0)

            print()

        except KeyboardInterrupt:
            print("\n中断")
            break
        except Exception as e:
            print(f"错误: {e}")

    client.close()
    print("已断开连接")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="交互式机器人命令工具")
    parser.add_argument("--ip", default=ROBOT_IP, help="机器人 IP 地址")
    parser.add_argument("--port", type=int, default=WS_PORT, help="WebSocket 端口")
    args = parser.parse_args()

    ROBOT_IP = args.ip
    WS_PORT = args.port

    main()
