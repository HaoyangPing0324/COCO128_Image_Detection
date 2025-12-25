"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : SSD训练配置
@License : MIT License (MIT)
@Version : 1.0
"""
import os
import torch

# ===================== 1. 设备配置（GPU/CPU自动适配） =====================
# 检查GPU是否可用，优先使用第0块GPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ===================== 2. 训练核心配置（用户指定参数） =====================
OUTPUT_DIR = "runs/detect"
DATA_ROOT = "./datasets/coco128"

# 训练超参数
BATCH_SIZE = 16       # GPU批次大小（CPU建议改为4）
IMG_SIZE = 300        # SSD固定输入尺寸，必须为300
EPOCHS = 15           # 训练总轮次（SSD收敛更快）
LEARNING_RATE = 0.001 # 初始学习率（适配SSD）
MODEL_TYPE = "ssd"    # 模型类型切换为ssd
Name = "ssd_test"     # 结果所在文件夹名称