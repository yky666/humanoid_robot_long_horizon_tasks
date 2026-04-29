import socket
import json

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"🎧 正在监听 UDP 端口 {UDP_PORT} ...")

while True:
    data, addr = sock.recvfrom(4096)
    msg = json.loads(data.decode('utf-8'))
    
    if 'right' in msg:
        # 读取右手矩阵的 X, Y, Z (第四列)
        x = msg['right'][0][3]
        y = msg['right'][1][3]
        z = msg['right'][2][3]
        print(f"\r收到来自 {addr[0]} 的数据 -> 右手坐标: X:{x:.3f} Y:{y:.3f} Z:{z:.3f}", end='')