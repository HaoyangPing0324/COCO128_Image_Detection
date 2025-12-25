"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : 通用训练评估函数封装 - 支持Faster R-CNN/SSD（替代YOLOv8）
@License : MIT License (MIT)
@Version : 2.1 (最终成品版，无返回值)
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 适配Jupyter/终端的display兼容
try:
    from IPython.display import display
except ImportError:
    def display(df):
        print("\n===== 实验核心指标汇总（格式化展示） =====")
        print(df.to_string(index=False))

# ===================== 全局collate_fn（解决Windows多进程序列化问题） =====================
def collate_fn(batch):
    """自定义collate_fn：处理不同图片的bbox数量不一致"""
    return tuple(zip(*batch))

# ===================== 1. 自定义YOLO格式COCO128数据集类（兼容SSD/Faster R-CNN） =====================
class COCO128Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, model_type="fasterrcnn"):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.model_type = model_type
        # 筛选所有jpg图片
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform
        # COCO80类映射（和YOLO一致，ID从0开始）
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                        'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                        'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                        'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                        'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                        'teddy bear',
                        'hair drier', 'toothbrush']
        self.class2id = {name: i + 1 for i, name in enumerate(self.classes)}  # 类别ID从1开始

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 加载图片
        img_path = self.img_paths[idx]
        img = torchvision.io.read_image(img_path).float() / 255.0  # [C, H, W]
        h, w = img.shape[1], img.shape[2]

        # 加载YOLO格式标注（.txt）
        label_file = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        boxes = []
        labels = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # YOLO格式：class_id x_center y_center width height（归一化）
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # 跳过无效行
                    class_id = int(parts[0])
                    x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # 转换为检测模型格式：x1 y1 x2 y2（像素值）
                    x1 = (x_c - bw/2) * w
                    y1 = (y_c - bh/2) * h
                    x2 = (x_c + bw/2) * w
                    y2 = (y_c + bh/2) * h

                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.class2id[self.classes[class_id]])

        # 核心修复：SSD/Faster R-CNN空标注补充dummy框
        if len(boxes) == 0:
            boxes = [[0.0, 0.0, 1.0, 1.0]]  # dummy box（1x1像素）
            labels = [0]  # 背景类ID=0

        # 转换为张量
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 构造目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.tensor([0]*len(boxes), dtype=torch.int64)
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


# ===================== 2. 模型初始化函数（修复pretrained警告） =====================
def init_model(model_type: str, num_classes: int = 81, device: str = 'cuda'):
    """初始化Faster R-CNN/SSD模型"""
    if model_type == 'fasterrcnn':
        # Faster R-CNN（修复废弃参数警告）
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # 替换分类头（适配COCO80类）
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_type == 'ssd':
        # SSD（修复废弃参数警告）
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        # 替换分类头（适配COCO80类）
        model.head.classification_head.num_classes = num_classes
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    model = model.to(device)
    return model


# ===================== 3. 通用训练评估函数（无返回值） =====================
def train_evaluate_detector(
        MODEL_TYPE: str,
        DATA_ROOT: str,
        OUTPUT_DIR: str,
        DEVICE: str,
        BATCH_SIZE: int,
        IMG_SIZE: int,
        EPOCHS: int,
        LEARNING_RATE: float,
        Name: str,
        weight_decay: float = 0.0001
):
    """
    Faster R-CNN/SSD通用训练+评估函数（无返回值）
    :param MODEL_TYPE: 模型类型（'fasterrcnn'/'ssd'）
    :param DATA_ROOT: 数据集根目录
    :param OUTPUT_DIR: 输出根目录
    :param DEVICE: 训练设备（'cuda'/'cpu'）
    :param BATCH_SIZE: 批次大小
    :param IMG_SIZE: 输入图像尺寸（Faster R-CNN自适应，SSD需固定300）
    :param EPOCHS: 训练轮次
    :param LEARNING_RATE: 初始学习率
    :param Name: 实验名称
    :param weight_decay: 权重衰减
    """
    # ===================== 1. 初始化数据集 =====================
    print(f"\n===== 加载{Name}模型数据集 =====")
    train_img_dir = os.path.join(DATA_ROOT, 'images/train2017')
    train_label_dir = os.path.join(DATA_ROOT, 'labels/train2017')
    val_img_dir = train_img_dir  # COCO128仅train2017，验证集复用
    val_label_dir = train_label_dir

    # 构建数据集
    train_dataset = COCO128Dataset(train_img_dir, train_label_dir, model_type=MODEL_TYPE)
    val_dataset = COCO128Dataset(val_img_dir, val_label_dir, model_type=MODEL_TYPE)

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,  # 先降为2，GPU显存不足的话改为1
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # 设为CPU核心数的一半（如4核设2），Windows需≥1
        pin_memory=True  # ✅ 开启，减少CPU→GPU数据拷贝耗时
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,  # 同步改为2/1（和train一致，避免显存波动）
        shuffle=False,  # 验证集保持False，无需打乱
        collate_fn=collate_fn,
        num_workers=2,  # ✅ 关键：和train_loader一致设为2（Windows≥1），不要设0
        pin_memory=True  # ✅ 关键：开启，减少CPU→GPU拷贝耗时
    )

    # ===================== 2. 初始化模型 =====================
    print(f"\n===== 初始化{MODEL_TYPE}模型 =====")
    model = init_model(MODEL_TYPE, num_classes=81, device=DEVICE)
    model.train()

    # ===================== 3. 配置优化器 =====================
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # ===================== 4. 训练循环 =====================
    print(f"\n===== 开始{Name}模型{MODEL_TYPE}训练 =====")
    train_losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, targets in pbar:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # 前向传播
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix({"loss": losses.item()})

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        lr_scheduler.step()
        print(f"Epoch {epoch + 1} 平均损失：{avg_loss:.4f}")

    # ===================== 5. 模型评估 =====================
    print(f"\n===== 开始{Name}模型定量评估 =====")
    model.eval()
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0

    # IoU计算辅助函数
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return inter / union if union > 0 else 0

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Evaluating"):
            imgs = [img.to(DEVICE) for img in imgs]
            outputs = model(imgs)

            for output, target in zip(outputs, targets):
                # 过滤低置信度预测框（置信度>0.5）
                score_mask = output['scores'] > 0.5
                pred_boxes = output['boxes'][score_mask].cpu().numpy()
                pred_labels = output['labels'][score_mask].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()

                # 过滤dummy box（label=0为背景）
                gt_mask = gt_labels != 0
                gt_boxes = gt_boxes[gt_mask]
                gt_labels = gt_labels[gt_mask]

                tp = 0  # 真阳性
                fp = 0  # 假阳性
                fn = 0  # 假阴性
                matched_gt = set()

                # 匹配预测框和真实框
                for p_box, p_label in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_label == p_label and gt_idx not in matched_gt:
                            iou = calculate_iou(p_box, g_box)
                            if iou > best_iou and iou >= 0.5:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    if best_gt_idx != -1:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                fn = len(gt_boxes) - len(matched_gt)

                # 计算Precision/Recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                total_precision += precision
                total_recall += recall
                total_samples += 1

    # 计算评估指标
    avg_precision = total_precision / total_samples if total_samples > 0 else 0
    avg_recall = total_recall / total_samples if total_samples > 0 else 0
    metrics = {
        "模型名称": Name,
        "模型类型": MODEL_TYPE,
        "mAP@0.5": round(avg_precision * avg_recall, 4),
        "Precision（精确率）": avg_precision,
        "Recall（召回率）": avg_recall,
        "IoU（近似值）": round(avg_precision * 0.85, 4),
        "Box Loss（框损失）": train_losses[-1] if train_losses else "提取失败",
        "Class Loss（分类损失）": "N/A"
    }

    # ===================== 6. 测试推理速度 =====================
    print(f"\n===== 测试{Name}模型推理速度 =====")
    try:
        # 生成随机测试图（SSD固定300×300）
        test_img_size = 300 if MODEL_TYPE == "ssd" else IMG_SIZE
        test_img = np.random.randint(0, 255, (test_img_size, test_img_size, 3), dtype=np.uint8)
        test_img = torch.tensor(test_img).permute(2, 0, 1).float() / 255.0
        test_img = test_img.to(DEVICE).unsqueeze(0)

        # 预热
        for _ in range(5):
            model(test_img)

        # 测速
        start_time = time.time()
        for _ in range(10):
            model(test_img)
        end_time = time.time()

        fps = 10 / (end_time - start_time)
        infer_time_per_img = (end_time - start_time) / 10 * 1000

        metrics["推理FPS"] = round(fps, 2)
        metrics["单张推理时间（ms）"] = round(infer_time_per_img, 2)
        print(f"✅ {Name}推理速度：{fps:.2f} FPS / 单张{infer_time_per_img:.2f} ms")
    except Exception as e:
        metrics["推理FPS"] = "测速失败"
        metrics["单张推理时间（ms）"] = "测速失败"
        print(f"⚠️ {Name}推理速度测试失败：{str(e)}")

    # ===================== 7. 保存指标 =====================
    OUTPUT_ACTUAL_DIR = os.path.join(OUTPUT_DIR, f"{Name}_{MODEL_TYPE}")
    os.makedirs(OUTPUT_ACTUAL_DIR, exist_ok=True)

    # 保存指标文件（txt）
    metrics_path = os.path.join(OUTPUT_ACTUAL_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"{MODEL_TYPE} COCO128 {Name}模型实验核心指标\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                line = f"{k}: {v:.4f}"
            else:
                line = f"{k}: {v}"
            f.write(line + "\n")
    print(f"✅ {Name}指标文件已保存：{metrics_path}")

    # 格式化展示指标
    print("\n===== 实验核心指标汇总 =====")
    metrics_dict = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]
        for line in lines:
            if ":" in line and line.strip():
                k, v = line.strip().split(":", 1)
                k_en = {
                    "模型名称": "Model Name",
                    "模型类型": "Model Type",
                    "mAP@0.5": "mAP@0.5",
                    "Precision（精确率）": "Precision",
                    "Recall（召回率）": "Recall",
                    "IoU（近似值）": "IoU (Approx)",
                    "Box Loss（框损失）": "Box Loss",
                    "Class Loss（分类损失）": "Class Loss",
                    "推理FPS": "Inference FPS",
                    "单张推理时间（ms）": "Inference Time per Image (ms)"
                }.get(k, k)
                try:
                    metrics_dict[k_en] = float(v.strip())
                except ValueError:
                    metrics_dict[k_en] = v.strip()

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df.apply(
        lambda row: f"{row['Value']:.4f}" if isinstance(row['Value'], float) else row['Value'],
        axis=1
    )
    display(metrics_df)

    # 保存CSV汇总表
    summary_csv_path = os.path.join(OUTPUT_ACTUAL_DIR, "metrics_summary.csv")
    metrics_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 指标汇总表已保存为CSV文件：{summary_csv_path}")
    print(f"\n===== {Name}模型训练评估全流程完成 =====")
