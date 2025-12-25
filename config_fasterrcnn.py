"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : Faster R-CNN训练配置
@License : MIT License (MIT)
@Version : 1.0
"""
import os
import torch

# ===================== 1. 设备配置（GPU/CPU自动适配） =====================
# 检查GPU是否可用，优先使用第0块GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # ✅ 正确

# ===================== 2. 训练核心配置（用户指定参数） =====================
OUTPUT_DIR = "runs/detect"
DATA_ROOT = "./datasets/coco128"

# 训练超参数
BATCH_SIZE = 8        # GPU批次大小（CPU建议改为2）
IMG_SIZE = 640        # Faster R-CNN自适应尺寸，无需固定
EPOCHS = 20           # 训练总轮次
LEARNING_RATE = 0.005 # 初始学习率（适配Faster R-CNN）
MODEL_TYPE = "fasterrcnn"  # 模型类型固定为fasterrcnn
Name = "fasterrcnn_test"   # 结果所在文件夹名称