import cv2
import time

def test_camera(index):
    print(f"\n==============================")
    print(f"🚀 正在测试设备节点: /dev/video{index}")
    
    # 强制传入纯数字 index，完美避开警告
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ 无法打开设备 {index}")
        return False
        
    # 强制告诉相机：给我 MJPG 格式，1080p
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("⏳ 等待大疆相机预热 (2秒)...")
    time.sleep(2.0) # 充分给足时间让大疆切换模式
    
    success = False
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"🎉 见证奇迹！设备 {index} 是真正的视频流！画面尺寸: {frame.shape}")
            success = True
            break
        print(f"   ... 第 {i+1} 次抓取失败，继续尝试...")
        time.sleep(0.5)
        
    if not success:
        print(f"❌ 设备 {index} 能打开，但抓不到画面 (可能是 Metadata 节点)。")
        
    cap.release()
    return success

# 连扫大疆的两个分身
test_camera(0)
test_camera(1)