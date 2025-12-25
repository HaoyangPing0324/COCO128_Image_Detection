# COCO128_Image_Detection
基于COCO128数据集的目标检测项目，以YOLOv8为基础完成全流程实验，并拓展多模型对比、超参数分析、轻量化及可视化等方向研究。

## 核心研究内容
### 1. 基础目标检测
- 基于YOLOv8实现COCO128数据集目标检测全流程（数据预处理→模型训练→测试评估）；
- 定量分析检测结果：mAP、IoU、Precision、Recall等核心指标；
- 多维度实验分析：模型结构、训练过程（损失函数/优化器/超参数）、结果评价（可视化/错误案例分析）。

### 2. 拓展实验
- 多模型对比：YOLOv8 / Faster R-CNN / SSD 性能、速度、效果差异；
- 超参数分析：输入图像尺寸、训练轮次、学习率对性能的影响；
- 模型轻量化：SlimYOLO、NanoYOLO 压缩/加速测试；
- 显著性可视化：Grad-CAM/Score-CAM 可视化网络目标关注区域。

## 文件结构
## 文件结构
```bash
COCO128_Image_Detection/
├─ 配置文件/脚本
│  ├─ config_*.py          # 训练参数配置（epochs/image_size/lr/模型）
│  ├─ yolov8_main_*.py     # YOLOv8训练/评估脚本（不同参数/模型）
│  ├─ fasterrcnn_main.py   # Faster R-CNN训练脚本
│  ├─ ssd_main.py          # SSD训练脚本
│  ├─ generate_lightweight_yolo.py  # 轻量化模型生成脚本
│  └─ yolov8_saliency_visualization.py  # 显著性图可视化脚本
│
├─ datasets/
│  └─ coco128/
│     ├─ images/train2017/  # COCO128训练图片
│     └─ labels/train2017/  # COCO128标注文件
│
├─ runs/detect/            # 训练结果/可视化输出
│  ├─ 不同训练任务目录/     # 如init_train/epochs_50_train/...
│  ├─ saliency_maps_all_objects/  # 显著性图结果
│  └─ *Comparison.xlsx     # 模型/参数对比表格
│
└─ 模型文件/
   ├─ yolov8n.pt           # 基线模型权重
   ├─ slimyolov8n.pt       # SlimYOLO权重
   └─ nanoyolov8.pt        # NanoYOLO权重
