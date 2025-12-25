"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : SSD模型执行脚本
@License : MIT License (MIT)
@Version : 1.0
"""
# ===================== 导入核心模块 =====================
# 导入自定义训练评估函数
import os
import torch
from othermodel_train_evaluate import train_evaluate_detector
from config_ssd import (
    DEVICE, OUTPUT_DIR, DATA_ROOT,
    BATCH_SIZE, IMG_SIZE, EPOCHS, LEARNING_RATE, MODEL_TYPE, Name
)

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # 调用函数
    train_evaluate_detector(
        MODEL_TYPE=MODEL_TYPE,
        DATA_ROOT=DATA_ROOT,
        OUTPUT_DIR=OUTPUT_DIR,
        DEVICE=DEVICE,
        BATCH_SIZE=BATCH_SIZE,
        IMG_SIZE=IMG_SIZE,
        EPOCHS=EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        Name=Name
    )
