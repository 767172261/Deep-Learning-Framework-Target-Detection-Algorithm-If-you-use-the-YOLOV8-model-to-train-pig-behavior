深度学习框架目标检测算法如果使用YOLOV8模型训练猪行为检测数据集，通过训练的权重推理识别检测猪睡觉 行走吃东西 躺卧等6类行为
@[toc]
### **猪行为检测YOLO数据集**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d218d22e07d440f88de38012746791d6.png)

#### **数据集概述**

| 项目 | 内容 |
|------|------|
| **数据集名称** | 猪行为检测YOLO数据集 |
| **总图像数量** | 11,862 张 |
| **训练集数量** | 9,489 张（80%） |
| **验证集数量** | 2,373 张（20%） |
| **类别数量** | 6 类 |
| **分类标签** | `'Lying'`, `'Sleeping'`, `'Investigating'`, `'Eating'`, `'Walking'`, `'Moutend'` |
| **数据格式** | YOLO 格式（.txt 标签文件） |
| **应用场景** | 智能养殖、动物行为分析、健康监测、自动化管理 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4eeddd0f7f148648ae42a6bdf2c4de2.png)

#### **类别详细统计表**

| 行为类别（英文） | 行为类别（中文） | 图片数量 | 标注框数量 | 训练集图片数（≈80%） | 验证集图片数（≈20%） |
|------------------|------------------|----------|-------------|------------------------|------------------------|
| Lying            | 躺卧             | 1,823    | 2,250       | 1,458                  | 365                    |
| Sleeping         | 睡觉             | 1,976    | 9,361       | 1,581                  | 395                    |
| Investigating    | 探究（探索行为） | 2,045    | 2,086       | 1,636                  | 409                    |
| Eating           | 进食             | 1,978    | 2,579       | 1,582                  | 396                    |
| Walking          | 行走             | 1,984    | 2,158       | 1,587                  | 397                    |
| Moutend          | 咬/咀嚼（口部动作） | 2,032  | 2,032       | 1,626                  | 406                    |
| **总计**         | ——               | **11,862** | **20,466** | **9,489**              | **2,373**              |


---



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce8f0190ec0d4f04bbb13bd8ecc60772.png)
使用 **YOLOv8** 进行训练和部署。



---

## 🛠️ 一、环境搭建（支持 GPU 加速）

### 1. 安装 NVIDIA 驱动和 CUDA

```bash
# 检查 GPU 是否可用
nvidia-smi
```

确保输出中显示您的 GPU 型号和 CUDA 版本（如 12.4）。若未安装，请前往：
👉 [NVIDIA 官网下载驱动](https://www.nvidia.com/Download/index.aspx)

---

### 2. 安装 Anaconda

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

安装后打开 **Anaconda Prompt**（Windows）或终端（Linux/macOS）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 pig_behavior_yolo 的环境
conda create -n pig_behavior_yolo python=3.10

# 激活环境
conda activate pig_behavior_yolo
```

---

### 4. 安装 PyTorch（支持 GPU）

推荐使用 CUDA 11.8 或 12.1：

```bash
# 示例：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证是否成功：

```python
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

### 5. 安装 YOLOv8 和其他依赖

```bash
# 安装 Ultralytics YOLOv8（官方最新版）
pip install ultralytics

# 安装 OpenCV 和可视化工具
pip install opencv-python opencv-python-headless matplotlib seaborn numpy pandas tqdm

# 日志与评估
pip install tensorboard scikit-learn
```

验证安装：

```bash
yolo version
# 输出应为：8.1.x 或更高
```

---

## 📁 二、数据集结构与 `data.yaml` 配置

### 文件结构（建议组织方式）

```
pig_behavior_dataset/
├── images/
│   ├── train/        # 9489 张
│   └── val/          # 2373 张
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

> ✅ 假设同学您已按 8:2 划分训练集与验证集。标签为 YOLO 格式（`.txt`），每行：`class_id x_center y_center width height`（归一化坐标）

---

### ✅ `data.yaml` 内容

```yaml
# data.yaml
train: ./pig_behavior_dataset/images/train
val: ./pig_behavior_dataset/images/val

# 类别数量
nc: 6

# 类别名称（顺序必须与 label 中的 class_id 一致）
names: 
  - Lying
  - Sleeping
  - Investigating
  - Eating
  - Walking
  - Moutend
```

> 💡 保存为 `pig_behavior_dataset/data.yaml`

---

## 🚀 三、训练代码：`train.py`

```python
# train.py
from ultralytics import YOLO
import torch

# ================== 配置参数 ==================
DATA_CONFIG = 'pig_behavior_dataset/data.yaml'      # 数据集配置文件
MODEL_NAME = 'yolov8m.pt'                           # 推荐 medium 模型（精度与速度平衡）
PROJECT = 'runs/pig_behavior'                       # 项目保存路径
EXPERIMENT_NAME = 'yolov8m_pig_action'              # 实验名称
EPOCHS = 150                                        # 行为识别建议充分训练
BATCH_SIZE = 32                                     # 根据显存调整（建议 ≥ 8GB GPU）
IMG_SIZE = 640                                      # 输入尺寸（可尝试 1280 提升小目标性能）
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ================ 加载预训练模型（迁移学习）===============
model = YOLO(MODEL_NAME)

# ================ 开始训练 ================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=EXPERIMENT_NAME,
    exist_ok=True,
    device=DEVICE,
    workers=8,
    optimizer='AdamW',              # 对复杂行为更稳定
    lr0=0.001,                      # 初始学习率
    lrf=0.1,                        # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    patience=30,                    # 早停：30轮无提升则停止
    save=True,
    save_period=10,
    cache=False,                    # 若内存充足可设为 True
    single_cls=False,
    close_mosaic=20,                # 最后20轮关闭 Mosaic
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5                      # 左右翻转增强
)

print("✅ 训练完成！")
print(f"最佳模型保存路径：{PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")
```

> 📌 训练完成后，最佳模型保存在：
> ```
> runs/pig_behavior/yolov8m_pig_action/weights/best.pt
> ```

---

## 🔍 四、推理代码：`detect.py`

```python
# detect.py
from ultralytics import YOLO
import cv2
import os

# ================== 配置参数 ==================
MODEL_PATH = 'runs/pig_behavior/yolov8m_pig_action/weights/best.pt'
SOURCE = 'demo_pig_farm.mp4'        # 可为图片、视频、摄像头、文件夹
CONF_THRESHOLD = 0.3                # 行为检测建议低阈值提高召回
IOU_THRESHOLD = 0.45
SHOW_RESULT = True
SAVE_RESULT = True
OUTPUT_DIR = 'runs/detect/pig_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================ 加载模型 ================
model = YOLO(MODEL_PATH)

# ================ 推理 ================
results = model.predict(
    source=SOURCE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    save=SAVE_RESULT,
    project=OUTPUT_DIR,
    name='detection',
    exist_ok=True,
    imgsz=IMG_SIZE,
    device=DEVICE,
    show=SHOW_RESULT,
    stream=True
)

# ================ 遍历结果 ================
for r in results:
    im_array = r.plot()
    if SHOW_RESULT:
        cv2.imshow('Pig Behavior Detection', im_array)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print(f"✅ 推理完成！结果保存在: {OUTPUT_DIR}")
```

> ✅ 支持输入类型：
> - 图片：`'pig_001.jpg'`
> - 视频：`'pen_footage.mp4'`
> - 摄像头：`0`
> - RTSP 流：`'rtsp://...'`

---

## 📊 五、评估代码：`val.py`

```python
# val.py
from ultralytics import YOLO

# ================== 配置参数 ==================
MODEL_PATH = 'runs/pig_behavior/yolov8m_pig_action/weights/best.pt'
DATA_CONFIG = 'pig_behavior_dataset/data.yaml'
IMG_SIZE = 640
BATCH_SIZE = 32
CONF_THRESHOLD = 0.001
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# ================ 加载模型 ================
model = YOLO(MODEL_PATH)

# ================ 评估 ================
metrics = model.val(
    data=DATA_CONFIG,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    conf=CONF_THRESHOLD,
    iou=0.6,
    device=DEVICE,
    plots=True,           # 生成 PR 曲线、混淆矩阵
    save_json=True        # 生成 predictions.json
)

# ================ 打印关键指标 ================
print("\n" + "="*60)
print("📊 猪行为检测模型评估结果")
print("="*60)
print(f"mAP@0.5:        {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {metrics.box.map:.4f}")
print(f"Precision:      {metrics.box.precision.mean():.4f}")
print(f"Recall:         {metrics.box.recall.mean():.4f}")
print(f"F1 Score:       {metrics.box.f1.mean():.4f}")

# 按类别输出 AP
class_names = ['Lying', 'Sleeping', 'Investigating', 'Eating', 'Walking', 'Moutend']
for i, name in enumerate(class_names):
    print(f"{name:12} AP50: {metrics.box.ap[i]:.4f}")

print("="*60)
```

---

以上文字及代码仅供参考学习使用。
