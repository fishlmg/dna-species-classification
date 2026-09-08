# DNA 序列物种分类

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

基于深度学习的 DNA 序列物种分类项目。通过对 GenBank FASTA 序列进行特征编码，使用 PyTorch 构建神经网络模型，实现仅凭 DNA 序列自动判别物种来源。

## 📖 项目背景

DNA 条形码（DNA Barcoding）是通过一段标准基因序列（如 COI 基因）鉴定物种的技术。传统方法依赖序列比对（BLAST），本项目探索用深度学习直接从序列中学习物种判别特征，对比了两种不同的序列表征思路：

- **AC 方法**（`AC/`）：将碱基序列 One-hot 编码后输入 1D-CNN / 全连接网络，让模型自动学习局部碱基模式
- **k-mer 方法**（`k_mer/`）：将序列切分为 k-mer 短片段（类似文本中的 n-gram），构建频率特征进行分类

## ✨ 特性

- 支持 FASTA 格式序列读取与预处理（Biopython）
- One-hot 编码支持简并碱基（N/W/S/M/K 等 IUPAC 模糊碱基）
- 完整的训练 → 测试 → 预测流程
- NumPy 预处理缓存（`.npy`），避免重复解析 FASTA

## 🗂 项目结构

```
.
├── AC/                          # One-hot + 神经网络方案
│   ├── data.py                  # FASTA 读取、编码、数据集生成
│   ├── data_test.py             # 数据管道测试
│   ├── train.py                 # 模型训练
│   ├── model_test.py            # 模型结构测试
│   ├── test.py / test1.py       # 模型评估
│   ├── predict.py               # 单序列推理预测
│   ├── AC.txt / ac1~3.txt       # GenBank 序列号清单（物种列表）
│   ├── data/                    # FASTA 原始序列（不入库）
│   └── numpy/                   # 预处理后的 .npy 缓存（不入库）
└── k_mer/
    └── data.py                  # k-mer 特征提取
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
#    将 FASTA 文件放入 AC/data/，序列号清单见 AC/AC.txt
python AC/data.py          # FASTA → 预处理 .npy

# 3. 训练模型
python AC/train.py

# 4. 评估
python AC/test.py

# 5. 预测新序列
python AC/predict.py
```

## 🧬 编码说明

One-hot 编码字典（含简并碱基）：

| 碱基 | 编码 | 含义 |
| --- | --- | --- |
| A / T / C / G | `[1,0,0,0]` 等 | 标准碱基 |
| N | `[0,0,0,0]` | 未知碱基 |
| W / S / M / K | 两个碱基各 0.5 | 简并碱基（如 W = A/T） |

## 📦 依赖

PyTorch、scikit-learn、NumPy、Biopython —— 详见 [requirements.txt](requirements.txt)

## 📝 说明

- 数据集（FASTA 序列，体积较大）与训练权重（`.pth`）不包含在仓库中，已通过 `.gitignore` 排除，仅保留在本地
- 代码以学习/实验为目的，模型结构与超参数可按需调整

## 📄 许可证

[MIT License](LICENSE)
