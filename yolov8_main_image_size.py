"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : YOLOv8n模型执行脚本 - 改变图像尺寸
@License : MIT License (MIT)
@Version : 1.0
"""
# ===================== 导入核心模块 =====================
# 导入封装的YOLOv8训练评估通用函数
from yolov8_train_evaluate import train_evaluate_yolov8

# 导入初始配置参数（YOLOv8n基础配置）
from config_image_size import (
    MODEL, DATA_YAML, OUTPUT_DIR, DEVICE,
    BATCH_SIZE, IMG_SIZE, EPOCHS, LEARNING_RATE, Name
)

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # 调用核心训练评估函数，执行YOLOv8n完整训练+评估流程
    train_evaluate_yolov8(
        MODEL=MODEL,
        DATA_YAML=DATA_YAML,
        OUTPUT_DIR=OUTPUT_DIR,
        DEVICE=DEVICE,
        BATCH_SIZE=BATCH_SIZE,
        IMG_SIZE=IMG_SIZE,
        EPOCHS=EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        Name=Name  # YOLOv8n的专属名称，如"yolov8n"
    )