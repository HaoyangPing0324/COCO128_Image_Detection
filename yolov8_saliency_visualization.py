"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/25
@Desc    : YOLOv8 显著性图可视化 - 全目标版
@License : MIT License (MIT)
@Version : 4.4 (全目标最终版)
"""
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")  # 屏蔽无关警告

# ===================== 配置项 =====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "runs/detect"
TRAIN_NAME = "init_train"
MODEL_PATH = os.path.join(OUTPUT_DIR, TRAIN_NAME, "weights/best.pt")
TEST_IMG_PATH = "./datasets/coco128/images/train2017/000000000575.jpg"
SAVE_DIR = os.path.join(OUTPUT_DIR, TRAIN_NAME, "saliency_maps_all_objects")
IMG_SIZE = 640
OVERLAY_ALPHA = 0.5  # 叠加图透明度
# 单色系色板（二选一）
CMAP = "Reds"  # 红系（浅红→深红）
# CMAP = "Greys"  # 灰系（浅灰→纯黑）

def generate_saliency_all_objects():
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载模型（评估模式）
    model = YOLO(MODEL_PATH).to(DEVICE)
    model.eval()

    # ===================== 1. 图片预处理 =====================
    if not os.path.exists(TEST_IMG_PATH):
        raise FileNotFoundError(f"测试图片不存在：{TEST_IMG_PATH}")

    img = cv2.imread(TEST_IMG_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    # 标准化+开启梯度
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True
    img_tensor.retain_grad()

    # ===================== 2. 提取所有检测目标 =====================
    results = model(img, imgsz=IMG_SIZE, verbose=False)
    if len(results[0].boxes) == 0:
        raise ValueError("图片中未检测到任何目标！")

    # 提取所有目标的类别ID和置信度
    all_boxes = results[0].boxes
    all_cls_idx = [int(b.cls.cpu().numpy().item()) for b in all_boxes]
    all_cls_names = [model.names[idx] for idx in all_cls_idx]
    all_conf = [b.conf.cpu().numpy().item() for b in all_boxes]
    all_xyxy = [list(map(int, b.xyxy[0].cpu().numpy())) for b in all_boxes]

    # ===================== 3. 梯度计算（多目标叠加） =====================
    # 前向传播
    outputs = model.model(img_tensor)
    # 求和所有检测目标的类别得分（多目标叠加）
    loss = 0
    for cls_idx in all_cls_idx:
        cls_scores = outputs[0][..., 5 + cls_idx]
        loss += cls_scores[0].mean()
    loss = loss / len(all_cls_idx)  # 平均所有目标的得分

    # 反向传播
    model.zero_grad()
    loss.backward()

    # ===================== 4. 显著性图生成（单色系+深浅增强） =====================
    # 提取梯度并处理
    grads = img_tensor.grad.data.squeeze().cpu().numpy()
    grads = np.transpose(grads, (1, 2, 0))
    grads = np.abs(grads)

    # 归一化+对比度拉伸+极值截断（增强深浅）
    grad_min, grad_max = grads.min(), grads.max()
    grads = (grads - grad_min) / (grad_max - grad_min + 1e-10)
    # 对比度拉伸
    alpha = 2
    grads = np.power(grads, 1/alpha)
    # 截断极值
    percentile_low = 5
    percentile_high = 95
    grad_low = np.percentile(grads, percentile_low)
    grad_high = np.percentile(grads, percentile_high)
    grads = np.clip(grads, grad_low, grad_high)
    grads = (grads - grad_low) / (grad_high - grad_low + 1e-10)

    # 生成单色系热力图
    heatmap = grads.mean(axis=2)
    heatmap_scaled = (heatmap * 255).astype(np.uint8)

    # ===================== 5. 可视化绘制（显示所有目标） =====================
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

    # 子图1：原图+所有检测框（不同颜色区分不同目标）
    axes[0].imshow(img_resized)
    # 定义颜色列表（循环使用，区分不同目标）
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for i, (xyxy, cls_name, conf) in enumerate(zip(all_xyxy, all_cls_names, all_conf)):
        x1, y1, x2, y2 = xyxy
        color = colors[i % len(colors)]  # 循环取色
        # 绘制检测框
        axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, color=color, linewidth=2))
        # 绘制类别+置信度
        axes[0].text(x1, y1-10, f"{cls_name} ({conf:.2f})",
                    color=color, fontsize=10, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    axes[0].set_title("原始图像 + 所有检测目标", fontsize=14, weight='bold')
    axes[0].axis("off")

    # 子图2：单色系热力图（多目标叠加）
    vmin = 20
    vmax = 180
    im = axes[1].imshow(heatmap_scaled, cmap=CMAP, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"多目标注意力热力图（{CMAP}系）", fontsize=14, weight='bold')
    axes[1].axis("off")
    # 色标
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
    cbar.set_label("激活强度（越深越关注）", fontsize=10)

    # 子图3：单色系叠加图（所有目标+热力图）
    heatmap_rgb = plt.cm.get_cmap(CMAP)(heatmap_scaled/255.0)[:, :, :3]
    overlay = (OVERLAY_ALPHA * heatmap_rgb) + ((1 - OVERLAY_ALPHA) * img_resized / 255.0)
    overlay = overlay / overlay.max()
    axes[2].imshow(overlay)
    # 叠加图上绘制所有检测框（简化版）
    for i, xyxy in enumerate(all_xyxy):
        x1, y1, x2, y2 = xyxy
        color = colors[i % len(colors)]
        axes[2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, color=color, linewidth=1.5))
    axes[2].set_title("多目标显著性叠加图", fontsize=14, weight='bold')
    axes[2].axis("off")

    # ===================== 6. 保存结果 =====================
    save_path = os.path.join(SAVE_DIR, f"saliency_all_objects_{os.path.basename(TEST_IMG_PATH)}")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='white')
    print(f"✅ 全目标显著性图已保存：{save_path}")
    print(f"📌 共检测到 {len(all_boxes)} 个目标：{', '.join(all_cls_names)}")
    plt.show()

if __name__ == "__main__":
    try:
        generate_saliency_all_objects()
    except Exception as e:
        print(f"❌ 执行出错：{str(e)}")