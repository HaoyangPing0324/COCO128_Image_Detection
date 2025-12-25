"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : YOLOv8训练基础配置 - 改变图像尺寸
@License : MIT License (MIT)
@Version : 1.0
"""
import torch

# ===================== 1. 设备配置（GPU/CPU自动适配） =====================
# 检查GPU是否可用，优先使用第0块GPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ===================== 2. 训练核心配置（用户指定参数） =====================
# 数据集配置文件（官方内置coco128.yaml）
DATA_YAML = "coco128.yaml"
# 训练结果输出目录
OUTPUT_DIR = "runs/detect"

# 训练超参数
BATCH_SIZE = 16
IMG_SIZE = 896         # 640→896（32的倍数，显存占用↑50%，训练速度↓30%，大目标检测精度↑明显）
EPOCHS = 20
LEARNING_RATE = 0.01
MODEL = "yolov8n.pt"
Name = "image_size_896"