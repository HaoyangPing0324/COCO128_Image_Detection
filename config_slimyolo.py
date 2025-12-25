"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : YOLOv8训练基础配置 - SlimYOLO
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
IMG_SIZE = 640         # 和原版保持一致，仅改模型，单变量对比
EPOCHS = 20            # 轮次不变，避免干扰
LEARNING_RATE = 0.01   # 学习率不变，保证公平
MODEL = "slimyolov8n.pt"  # SlimYOLO轻量版（需先下载/配置）
Name = "model_slimyolo"   # 标识轻量模型