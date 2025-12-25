"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : SlimYOLO和NanoYOLO的生成
@License : MIT License (MIT)
@Version : 1.0
"""
from ultralytics import YOLO
import torch


def generate_slim_nano_yolo():
    # 1. 加载原版YOLOv8n
    original_model = YOLO("yolov8n.pt")

    # 2. 生成SlimYOLO并保存
    slim_model = YOLO("yolov8n.pt")
    for m in slim_model.model.modules():
        if hasattr(m, 'channels') and m.channels > 32:
            m.channels = int(m.channels * 0.5)
        if hasattr(m, 'in_channels') and m.in_channels > 32:
            m.in_channels = int(m.in_channels * 0.5)
    slim_model.save("slimyolov8n.pt")  # 保存到本地
    print("✅ SlimYOLO权重已保存：slimyolov8n.pt")

    # 3. 生成NanoYOLO并保存
    nano_model = YOLO("yolov8n.pt")
    for m in nano_model.model.modules():
        if hasattr(m, 'channels') and m.channels > 32:
            m.channels = int(m.channels * 0.25)
        if hasattr(m, 'in_channels') and m.in_channels > 32:
            m.in_channels = int(m.in_channels * 0.25)
    nano_model.save("nanoyolov8.pt")  # 保存到本地
    print("✅ NanoYOLO权重已保存：nanoyolov8.pt")


if __name__ == "__main__":
    generate_slim_nano_yolo()