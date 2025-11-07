# 钢筋数量计数系统（基于 YOLO 模型）
点个star吧求求了

该项目基于 YOLO 模型实现钢筋数量的自动检测与计数，通过数据增强、模型训练、预测及后处理等流程，实现对钢筋图像中目标的精准识别和数量统计。

## 项目功能

* 对钢筋图像进行数据增强，扩充训练样本
* 划分训练集、测试集和验证集，用于模型训练与评估
* 使用 YOLO 模型训练钢筋检测模型
* 对新图像进行钢筋检测预测
* 对预测结果进行后处理（筛选、去重、标记），输出最终计数结果

## 文件结构及说明

| 文件名称               | 功能描述                                                                         |
| ---------------------- | -------------------------------------------------------------------------------- |
| `mypredict.py`       | 加载训练好的 YOLO 模型，对指定文件夹中的图像进行钢筋检测预测，保存结果和标签文件 |
| `mytrain.py`         | 基于 YOLO 预训练模型，使用自定义数据集训练钢筋检测模型                           |
| `post_progress.py`   | 对预测生成的标签文件进行后处理（筛选下半部分目标、去除过近点、标记结果）         |
| `add_more_sample.py` | 对原始图像和标签进行数据增强（旋转、切割、缩放等），扩充训练样本                 |
| `split.py`           | 将增强后的数据集划分为训练集、测试集和验证集，用于模型训练                       |

## 环境依赖

* Python 3.8+
* 第三方库：
  
  * `ultralytics`（YOLO 模型库）
  * `Pillow`（图像处理）
  * 标准库：`os`、`math`、`shutil`、`random`等

安装依赖：

```
pip install ultralytics pillow
```

## 使用步骤

### 1. 数据准备与增强

1. 将原始钢筋图像（.bmp 格式）放入 `input_images_dir`，对应 YOLO 格式标签（.txt）放入 `input_labels_dir`
2. 修改 `add_more_sample.py`中的路径参数：

```
input\_images\_dir = "你的原始图像路径"

input\_labels\_dir = "你的原始标签路径"

output\_base = "增强后数据保存路径"
```

1. 运行数据增强脚本：

```
python add\_more\_sample.py
```

增强方式包括：旋转（90°/180°/270°）、切割（去除上方 20%/40%/60% 区域）、缩放（0.8x/1.2x/1.6x）

### 2. 划分数据集

1. 修改 `split.py`中的路径和划分比例：

```
image\_directory = "增强后的图像路径"  # 对应add\_more\_sample的output\_base/images

label\_directory = "增强后的标签路径"  # 对应add\_more\_sample的output\_base/labels

output\_directory = "划分后数据集保存路径"

train\_ratio = 0.7  # 训练集比例

test\_ratio = 0.2   # 测试集比例

val\_ratio = 0.1    # 验证集比例
```

1. 运行划分脚本：

```
python split.py
```

生成的数据集结构为：

```
output\_directory/

├─ images/

│  ├─ train/

│  ├─ test/

│  └─ val/

└─ labels/

   ├─ train/

   ├─ test/

   └─ val/
```

### 3. 模型训练

1. 准备数据集配置文件（如 `steel.yaml`），指定训练 / 验证集路径和类别信息
2. 修改 `mytrain.py`中的参数：

```
model = YOLO('yolo11n.pt')  # 选择YOLO预训练模型

data='steel.yaml路径'       # 数据集配置文件路径

epochs=100                 # 训练轮次

imgsz=640                  # 输入图像尺寸

batch=16                   # 批次大小
```

1. 运行训练脚本：

```
python mytrain.py
```

训练结果（包括最佳模型）保存在 `runs/detect/train*/weights/``best.pt`

### 4. 钢筋检测预测

1. 修改 `mypredict.py`中的参数：

```
model = YOLO('训练好的模型路径')  # 如runs/detect/train2/weights/best.pt

source='待检测图像文件夹路径'    # 输入图像路径

imgsz=640                     # 预测图像尺寸

conf=0.35                     # 置信度阈值
```

1. 运行预测脚本：

```
python mypredict.py
```

预测结果图像和标签文件保存在 `runs/detect/predict/`

### 5. 结果后处理

1. 修改 `post_progress.py`中的路径和后处理参数：

```
img\_dir = "待处理图像文件夹路径"        # 与预测的source一致

label\_dir = "预测生成的标签文件夹路径"  # 如runs/detect/predict/labels

output\_dir = "标记结果保存路径"

distance\_threshold = 20  # 去重距离阈值（像素）

min\_y\_ratio = 0.5        # 保留图像下半部分（y>0.5）的目标
```

1. 运行后处理脚本：

```
python post\_progress.py
```

处理后的标记图像（含钢筋中心点标记）保存在 `output_dir`，可直接查看计数结果

## 关键参数说明

| 参数名称               | 作用说明                   | 可调范围                  |
| ---------------------- | -------------------------- | ------------------------- |
| `epochs`             | 模型训练轮次               | 50-300（根据数据集大小）  |
| `imgsz`              | 图像尺寸（训练 / 预测）    | 416-1024（建议 640）      |
| `conf`               | 预测置信度阈值             | 0.2-0.5（过滤低置信目标） |
| `distance_threshold` | 去重距离阈值（像素）       | 10-50（根据钢筋密度调整） |
| `min_y_ratio`        | 保留区域比例（y 轴相对值） | 0-1（按需筛选目标区域）   |

## 注意事项

1. 确保所有文件路径使用绝对路径或正确的相对路径，避免路径错误
2. 原始标签文件需符合 YOLO 格式（class\_id x\_center y\_center width height，坐标为相对值）
3. 数据增强和训练过程可能耗时较长，建议根据硬件配置调整参数（如 `batch`大小）
4. 若检测效果不佳，可尝试增加训练数据、调整 `epochs`或更换更大的 YOLO 模型（如 yolo11m.pt）

