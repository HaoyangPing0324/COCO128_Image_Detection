"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : YOLOv8训练基础配置 - 初始配置
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
BATCH_SIZE = 16        # GPU批次大小（CPU建议改为8）
IMG_SIZE = 640         # 输入图像尺寸（YOLO系列需为32的倍数）
EPOCHS = 20            # 训练总轮次
LEARNING_RATE = 0.01   # 初始学习率
MODEL = "yolov8n.pt"   # 预训练模型权重文件
Name = "init"          # 结果所在文件夹名称