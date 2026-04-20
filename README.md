# HW1: 从零构建三层神经网络分类器 —— EuroSAT 地表覆盖分类

> 《深度学习与空间智能》课程作业。在**不使用任何深度学习框架**(PyTorch / TensorFlow / JAX)的前提下,仅用 NumPy 手工实现三层 MLP 的前向传播、反向传播与参数更新,完成 EuroSAT RGB 卫星图像的 10 类地表覆盖分类。

---

## 📋 项目概览

- **数据集**: EuroSAT RGB (10 类,共 27,000 张 64×64 卫星图像)
- **模型**: 三层全连接神经网络 (Input → FC1 → Act → FC2 → Act → FC3 → Softmax)
- **框架约束**: 仅使用 NumPy,手动实现反向传播
- **最终测试准确率**: 约 65%

---

## 🗂 目录结构

```
.
├── README.md                      # 本文件
├── requirements.txt               # 环境依赖
├── hw1.ipynb                      # 完整实验 Notebook (推荐入口)
├── src/                           # 模块化 Python 源码 (可选)
│   ├── data_loader.py             # 数据加载与预处理
│   ├── model.py                   # 三层 MLP 模型
│   ├── layers.py                  # 激活函数 + Softmax-CE 损失
│   ├── optimizer.py               # SGD 优化器与学习率调度
│   ├── trainer.py                 # 训练循环
│   ├── search.py                  # 网格超参数搜索
│   └── utils.py                   # 可视化工具
├── results/                       # 实验产出的图片
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── hyperparam_search.png
│   ├── weight_visualization.png
│   └── error_analysis.png
└── checkpoints/                   # 模型权重 (运行后生成, 较大不入库)
    └── best_model_final.pkl
```

> **数据集目录 `EuroSAT_RGB/` 请自行下载后放在项目根目录下,不在本仓库跟踪。**

---

## 🔗 模型权重下载

训练好的最优模型权重 (`best_model_final.pkl`) 托管在 Google Drive:

**📥 [点此下载 best_model_final.pkl](在此填入你的 Google Drive 分享链接)**

下载后请放到 `checkpoints/` 目录,测试脚本会自动加载。

---

## ⚙️ 环境依赖

- Python ≥ 3.8
- 推荐使用 conda 或 venv 隔离环境

### requirements.txt

```
numpy>=1.21
matplotlib>=3.4
Pillow>=9.0
jupyter>=1.0       # 运行 notebook 需要
```

### 快速安装

```bash
# 方法 1: pip
pip install -r requirements.txt

# 方法 2: conda
conda create -n hw1 python=3.10
conda activate hw1
pip install -r requirements.txt
```

---

## 📥 数据准备

1. 从 [EuroSAT 官方仓库](https://github.com/phelber/EuroSAT) 下载 `EuroSAT_RGB.zip`
2. 解压后得到 `EuroSAT_RGB/` 目录,内含 10 个子文件夹 (AnnualCrop、Forest、…、SeaLake)
3. 将 `EuroSAT_RGB/` 放在项目根目录下,或在代码中修改 `data_dir` 路径

数据目录结构应如下:
```
EuroSAT_RGB/
├── AnnualCrop/     (3000 张 jpg)
├── Forest/         (3000 张 jpg)
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

## 🚀 运行方式

### 直接跑 Jupyter Notebook

```bash
jupyter notebook hw1.ipynb
```

Notebook 从头到尾按顺序执行即可完成:数据加载 → 模型定义 → 训练 → 测试 → 可视化 → 超参搜索。

## 🧪 模型与超参数

### 网络结构
```
Input (12288) → Linear (hidden_dim) → ReLU
              → Linear (hidden_dim) → ReLU
              → Linear (10) → Softmax
```

### 最优超参数 (经 27 组网格搜索得到)

| 超参数 | 取值 |
|--------|------|
| 激活函数 | ReLU |
| 隐藏层维度 `h` | 512 |
| 初始学习率 `η` | 0.01 |
| 权重衰减 `λ` | 0 |
| Batch size | 128 |
| Epoch 数 | 80 |
| 学习率调度 | Step Decay (`step_size=25, γ=0.5`) |
| 权重初始化 | He |

### 训练/验证/测试划分

70% / 15% / 15% 分层采样,保证各类别比例均衡。
- 训练集: 18,900 张
- 验证集: 4,050 张
- 测试集: 4,050 张

---

## 📊 实验结果

### 测试集整体准确率: **~65%**

### 各类别准确率
| 类别 | 准确率 |
|------|--------|
| Forest | 88.7% |
| Pasture | 80.3% |
| SeaLake | 78.4% |
| Industrial | 77.1% |
| AnnualCrop | 63.1% |
| River | 61.1% |
| Residential | 56.9% |
| HerbaceousVegetation | 56.2% |
| PermanentCrop | 47.5% |
| Highway | 42.4% |

混淆矩阵、训练曲线、权重可视化、错例分析详见 `results/` 文件夹及实验报告。

---

## 📝 实现要点

- ✅ **反向传播**:手动推导链式法则,逐层实现梯度
- ✅ **数值稳定**:Softmax 减最大值、Sigmoid 分支实现防溢出
- ✅ **损失合并**:Softmax + CrossEntropy 合并计算,梯度形式简洁为 `(p - y_onehot) / N`
- ✅ **激活函数**:支持 ReLU / Sigmoid / Tanh 切换,配套 He / Xavier 初始化
- ✅ **SGD + L2**:权重衰减直接加入梯度 `W ← W - η(dW + λW)`,偏置不正则化
- ✅ **学习率调度**:Step Decay 与 Cosine Annealing 均可选
- ✅ **最优模型保存**:基于验证集准确率自动保存最佳权重
- ✅ **超参搜索**:网格搜索 27 组组合 (lr × hidden × wd)

---

## 📖 实验报告

实验报告 PDF 见仓库根目录或 [此处](./report.pdf)。

---

## ⚠️ 已知局限

- MLP 将图像展平为一维向量,**丢失了空间结构信息**,第一层权重可视化呈彩色噪声状
- 训练集 97% vs 验证集 66.5% 的差距反映 MLP 容量过大但特征提取能力不足
- 未来引入 CNN 预期可显著提升精度



---

## 📜 License

本项目仅用于《深度学习与空间智能》课程作业,请勿用于商业用途。
