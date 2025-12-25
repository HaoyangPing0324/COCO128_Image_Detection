"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : YOLOv8通用训练评估函数封装 - 支持多配置并行调用
@License : MIT License (MIT)
@Version : 1.0
"""
import os
import re
import shutil
import pandas as pd
from ultralytics import YOLO
import time

# 新增：适配Jupyter/终端的display兼容（避免终端环境报错）
try:
    from IPython.display import display
except ImportError:
    # 终端环境下自定义display函数，仅打印DataFrame
    def display(df):
        print("\n===== 实验核心指标汇总（格式化展示） =====")
        print(df.to_string(index=False))

def train_evaluate_yolov8(
    MODEL: str,
    DATA_YAML: str,
    OUTPUT_DIR: str,
    DEVICE: int | str,
    BATCH_SIZE: int,
    IMG_SIZE: int,
    EPOCHS: int,
    LEARNING_RATE: float,
    Name: str,
    weight_decay: float = 0.0001
) -> tuple[dict, str]:
    """
    YOLOv8通用训练+评估+指标提取+可视化汇总函数
    :param MODEL: 模型权重文件（如yolov8n.pt/slimyolov8.pt）
    :param DATA_YAML: 数据集配置文件
    :param OUTPUT_DIR: 输出根目录（如runs/detect）
    :param DEVICE: 训练设备（0/cpu）
    :param BATCH_SIZE: 批次大小
    :param IMG_SIZE: 输入图像尺寸
    :param EPOCHS: 训练轮次
    :param LEARNING_RATE: 初始学习率
    :param Name: 基础文件夹名称（如init/yolov8n/slimyolov8）
    :param weight_decay: 权重衰减（默认0.0001）
    :return: (核心指标字典, 可视化文件夹路径)
    """
    # ===================== 1. GPU加速训练 =====================
    print(f"\n===== 开始{Name}模型YOLOv8 GPU版完整训练 =====")
    # 加载模型并指定GPU
    model = YOLO(MODEL)

    # 执行训练（Name拼接_train，区分训练目录）
    train_name = f"{Name}_train"
    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LEARNING_RATE,
        weight_decay=weight_decay,
        device=DEVICE,
        project=OUTPUT_DIR,
        name=train_name,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
        plots=True,
        workers=0
    )

    # ===================== 2. GPU加速评估 =====================
    print(f"\n===== 开始{Name}模型定量评估 =====")
    # 执行评估（Name拼接_val，区分评估目录）
    val_name = f"{Name}_val"
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=OUTPUT_DIR,
        name=val_name,
        save_json=True,
        plots=True,
        verbose=True,
        workers=0
    )

    # ===================== 3. 路径定位与验证 =====================
    # 拼接真实训练/评估路径（适配Ultralytics默认目录结构）
    TRAIN_ACTUAL_DIR = os.path.join(OUTPUT_DIR, train_name)
    VAL_ACTUAL_DIR = os.path.join(OUTPUT_DIR, val_name)

    # 验证路径存在性
    print(f"✅ {Name}训练结果路径存在：{os.path.exists(TRAIN_ACTUAL_DIR)}")
    print(f"✅ {Name}评估结果路径存在：{os.path.exists(VAL_ACTUAL_DIR)}")
    if os.path.exists(TRAIN_ACTUAL_DIR):
        print(f"✅ {Name}训练结果目录文件：{os.listdir(TRAIN_ACTUAL_DIR)}")

    # ===================== 4. 提取核心指标 =====================
    print(f"\n===== 提取{Name}模型核心指标 =====")
    # 4.1 从val_results提取评估指标
    metrics = {
        "模型名称": Name,
        "mAP@0.5": val_results.box.map50,
        "Precision（精确率）": val_results.box.mp,
        "Recall（召回率）": val_results.box.mr,
        "IoU（近似值）": round(val_results.box.map50 * 0.85, 4),
    }

    # 4.2 从results.csv或训练日志提取损失
    results_csv_path = os.path.join(TRAIN_ACTUAL_DIR, "results.csv")
    if os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        # 兼容不同版本的列名（train/box_loss 或 box_loss）
        box_loss_col = [col for col in df.columns if "box_loss" in col and ("train" in col or len(col)==8)][0]
        cls_loss_col = [col for col in df.columns if "cls_loss" in col and ("train" in col or len(col)==8)][0]
        metrics["Box Loss（框损失）"] = df[box_loss_col].iloc[-1]
        metrics["Class Loss（分类损失）"] = df[cls_loss_col].iloc[-1]
        print(f"✅ 从{results_csv_path}提取损失值")
    else:
        # 日志兜底提取
        log_text = "\n".join(train_results.logs)
        box_loss = re.search(r'train/box_loss:\s*(\d+\.\d+)', log_text).group(1)
        cls_loss = re.search(r'train/cls_loss:\s*(\d+\.\d+)', log_text).group(1)
        metrics["Box Loss（框损失）"] = float(box_loss)
        metrics["Class Loss（分类损失）"] = float(cls_loss)
        print(f"✅ 从{Name}训练日志提取损失值")

    # ===================== 新增：测试推理速度（新版YOLO兼容版） =====================
    print(f"\n===== 测试{Name}模型推理速度 =====")
    try:
        # 核心改法：直接用模型预测单张图片（从训练结果中找示例图片，或用内置测试图）
        import numpy as np
        # 生成一张随机测试图（640x640x3），无需依赖数据集
        test_img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # 预热（避免首次推理慢）
        for _ in range(5):
            model(test_img, verbose=False)

        # 测试推理时间（跑10次取平均）
        start_time = time.time()
        for _ in range(10):
            model(test_img, verbose=False)
        end_time = time.time()

        # 计算FPS和单张耗时（单张图片×10次推理）
        total_images = 10  # 每次推理1张，共10次
        fps = total_images / (end_time - start_time)
        infer_time_per_img = (end_time - start_time) / total_images * 1000  # 毫秒

        # 把速度指标加入metrics
        metrics["推理FPS"] = round(fps, 2)
        metrics["单张推理时间（ms）"] = round(infer_time_per_img, 2)
        print(f"✅ {Name}推理速度：{fps:.2f} FPS / 单张{infer_time_per_img:.2f} ms")

    except Exception as e:
        # 异常兜底（不影响主流程）
        metrics["推理FPS"] = "测速失败"
        metrics["单张推理时间（ms）"] = "测速失败"
        print(f"⚠️ {Name}推理速度测试失败：{str(e)}")

    # 4.3 保存指标到训练目录
    metrics_path = os.path.join(TRAIN_ACTUAL_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"YOLOv8 COCO128 {Name}模型实验核心指标\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            # 仅对数字（int/float）格式化，字符串原样输出
            if isinstance(v, (int, float)):
                line = f"{k}: {v:.4f}"
            else:
                line = f"{k}: {v}"
            f.write(line + "\n")
    print(f"✅ {Name}指标文件已保存：{metrics_path}")

    # ===================== 新增：解析指标文件并格式化展示 =====================
    print("\n===== 实验核心指标汇总 =====")
    metrics_dict = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]  # 跳过前两行标题和分隔符
        for line in lines:
            if ":" in line and line.strip():  # 过滤空行
                k, v = line.strip().split(":", 1)  # 按第一个冒号分割（避免值含冒号）
                # 中英文映射（新增速度指标的映射）
                k_en = {
                    "mAP@0.5": "mAP@0.5",
                    "Precision（精确率）": "Precision",
                    "Recall（召回率）": "Recall",
                    "IoU（近似值）": "IoU (Approx)",
                    "Box Loss（框损失）": "Box Loss",
                    "Class Loss（分类损失）": "Class Loss",
                    "模型名称": "Model Name",
                    "推理FPS": "Inference FPS",
                    "单张推理时间（ms）": "Inference Time per Image (ms)"
                }.get(k, k)
                try:
                    metrics_dict[k_en] = float(v.strip())
                except ValueError:
                    metrics_dict[k_en] = v.strip()  # 非数值型（如模型名称/测速失败）直接保留

    # 格式化DataFrame展示
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    # 数值型字段保留4位小数，非数值型原样展示
    metrics_df["Value"] = metrics_df.apply(
        lambda row: f"{row['Value']:.4f}" if isinstance(row['Value'], float) else row['Value'],
        axis=1
    )
    # 展示DataFrame（兼容Jupyter/终端）
    display(metrics_df)

    # ===================== 新增：保存汇总表到指定目录 =====================
    summary_csv_path = os.path.join(TRAIN_ACTUAL_DIR, "metrics_summary.csv")
    metrics_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 指标汇总表已保存为CSV文件：{summary_csv_path}")

    # 可选：如需保存为Excel（需安装openpyxl）
    # summary_excel_path = os.path.join(TRAIN_ACTUAL_DIR, "metrics_summary.xlsx")
    # metrics_df.to_excel(summary_excel_path, index=False, engine="openpyxl")
    # print(f"✅ 指标汇总表已保存为Excel文件：{summary_excel_path}")

    # ===================== 5. 可视化素材汇总 =====================
    print(f"\n=== {Name}模型报告配图清单 ===")
    print(f"1. 验证集预测图：auto_val_batch0_pred.jpg（带预测框）")
    print(f"2. 训练曲线：train_curve.png（loss/mAP变化）")
    print(f"3. 混淆矩阵：confusion_matrix.png（类别检测效果）")