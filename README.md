# HW1: 从零构建三层神经网络分类器 —— EuroSAT 地表覆盖图像分类

> **课程**：EIE60003 大数据技术前沿讲座
> **约束**：不使用 PyTorch / TensorFlow / JAX 等自动微分框架，仅基于 NumPy 手工实现前向传播、反向传播与参数更新。

---

## 项目简介

本项目在 EuroSAT RGB 遥感图像数据集上，从零实现了一个三层全连接神经网络（MLP）分类器，完成 10 种地表覆盖类别的图像分类任务。

**最优模型测试集准确率：66.22%**（最优超参数：学习率 η=0.01，隐藏层维度 h=512，L2 正则化系数 λ=0）

---

## 数据集

使用 [EuroSAT](https://github.com/phelber/EuroSAT) RGB 子集，包含 10 个地表覆盖类别，共 27,000 张 64×64 像素卫星图像：


| 类别                 | 中文含义     | 样本数 |
| -------------------- | ------------ | ------ |
| AnnualCrop           | 一年生农作物 | 3,000  |
| Forest               | 森林         | 3,000  |
| HerbaceousVegetation | 草本植被     | 3,000  |
| Highway              | 高速公路     | 2,500  |
| Industrial           | 工业区       | 2,500  |
| Pasture              | 牧场         | 2,000  |
| PermanentCrop        | 多年生农作物 | 2,500  |
| Residential          | 住宅区       | 3,000  |
| River                | 河流         | 2,500  |
| SeaLake              | 海洋/湖泊    | 3,000  |

请将数据集解压至项目根目录下的 `EuroSAT_RGB/` 文件夹，目录结构如下：

```
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

---

## 环境依赖

Python 3.8+，仅需以下标准库与科学计算库：

```
numpy
Pillow
matplotlib
```

安装方式：

```bash
pip install numpy Pillow matplotlib
```

---

## 项目结构

```
.
├── hw1_eurosat_mlp.ipynb   # 主 Notebook，包含全部五个模块
├── best_model_final.pkl    # 训练好的最优模型权重（见下方下载地址）
├── README.md
└── EuroSAT_RGB/            # 数据集目录（需自行下载）
```

Notebook 内部模块划分：


| 模块                 | 内容                                                                              |
| -------------------- | --------------------------------------------------------------------------------- |
| 一、数据加载与预处理 | 图像读取、归一化、展平、分层划分（7:1.5:1.5）、z-score 标准化                     |
| 二、模型定义         | 三层 MLP，支持 ReLU / Sigmoid / Tanh，He / Xavier 初始化，手写反向传播            |
| 三、训练循环         | Mini-batch SGD、L2 正则化、Step Decay / Cosine Annealing 学习率调度、最优权重保存 |
| 四、超参数搜索       | 网格搜索（学习率 × 隐藏层维度 × 正则化系数，共 27 组）                          |
| 五、测试评估         | 混淆矩阵、各类准确率、权重可视化、错例分析                                        |

---

## 运行方式

### 训练

在 Jupyter 环境中按顺序运行 `hw1_eurosat_mlp.ipynb` 中的全部 Cell 即可完成数据加载、超参数搜索与最优模型训练。训练完成后，最优模型权重将自动保存为 `best_model_final.pkl`。

```bash
jupyter notebook hw1_eurosat_mlp.ipynb
```

### 测试（加载已有权重）

在 Notebook 的「测试评估」模块中，将 `best_model_final.pkl` 放置于与 Notebook 相同目录下，直接运行对应 Cell 即可输出：

* 测试集整体准确率
* 各类别分类准确率
* 混淆矩阵可视化
* 第一层权重可视化
* 错例分析图

---

## 模型结构

```
Input (12288) → Linear1 → ReLU → Linear2 → ReLU → Linear3 → Softmax (10类)
```

* 参数初始化：ReLU 使用 He 初始化，Sigmoid/Tanh 使用 Xavier 初始化
* 优化器：带 L2 权重衰减的 SGD
* 学习率调度：Step Decay（每 25 epoch 衰减为原来的 0.5 倍）
* 批量大小：128

---

## 实验结果

**各类别测试准确率：**


| 类别                 | 准确率 |
| -------------------- | ------ |
| Forest               | 88.67% |
| Pasture              | 80.33% |
| Industrial           | 77.07% |
| SeaLake              | 78.44% |
| AnnualCrop           | 63.11% |
| River                | 61.07% |
| Residential          | 56.89% |
| HerbaceousVegetation | 56.22% |
| PermanentCrop        | 47.47% |
| Highway              | 42.40% |

---

## 模型权重下载

训练好的最优模型权重（`best_model_final.pkl`）托管于 Google Drive：

🔗 [点击下载模型权重](https://drive.google.com/drive/folders/1D5ngYKDMrMDC6_6uPZTkv_tBWgBszg3V?usp=drive_link)

---

## 注意事项

* 本项目为个人作业，严禁使用 PyTorch、TensorFlow、JAX 等支持自动微分的框架
* 数据集不包含在本仓库中，请自行从 [EuroSAT 官方](https://github.com/phelber/EuroSAT) 获取 RGB 版本
