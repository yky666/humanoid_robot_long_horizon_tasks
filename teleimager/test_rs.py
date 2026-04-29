import pyrealsense2 as rs
import time

print("--- RealSense 硬件底层探测器 ---")
try:
    ctx = rs.context()
    if len(ctx.devices) > 0:
        print("正在发送物理电击重启相机，请等待 3 秒...")
        ctx.devices[0].hardware_reset()
        time.sleep(3)
except Exception as e:
    pass

def test_cam(fps, enable_depth=False):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, fps)
    if enable_depth:
        # 官方的 Depth 尺寸是 640x480
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    
    mode = "彩色 + 深度 (双开)" if enable_depth else "仅彩色"
    print(f"\n[测试] 尝试分辨率 848x480 @ {fps} FPS [{mode}]...")
    try:
        pipeline.start(config)
        # 连抽 5 帧，等曝光稳定
        for _ in range(5):
            frames = pipeline.wait_for_frames(5000)
        if frames.get_color_frame():
            print(f"==========> ✅ 测试成功！相机满血复活！这就是正确的配置！ <==========")
        pipeline.stop()
        return True
    except Exception as e:
        print(f"❌ 测试失败，相机没出图: {e}")
        return False

# 穷举 4 种官方可能的组合
if not test_cam(60, False):
    if not test_cam(30, False):
        if not test_cam(60, True):
            test_cam(30, True)
