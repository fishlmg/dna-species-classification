# DNA 序列物种分类

基于 PyTorch 的 DNA 序列物种分类项目，通过对 DNA FASTA 序列进行特征编码与深度学习建模，实现物种自动识别。

## 方法

- **AC 方法**（`AC/`）：将碱基序列 One-hot 编码后，使用 1D-CNN / 全连接网络进行分类
- **k-mer 方法**（`k_mer/`）：将序列切分为 k-mer 片段进行特征表示

## 文件结构

```
AC/
├── data.py / data_test.py   # 数据加载与预处理（FASTA → npy）
├── train.py                 # 模型训练
├── test.py / test1.py       # 模型测试
├── model_test.py            # 模型结构测试
├── predict.py               # 推理预测
├── AC.txt / ac1~3.txt       # GenBank 序列号清单
└── data/ numpy/             # FASTA 原始数据与预处理结果（不入库，见 .gitignore）
k_mer/
└── data.py                  # k-mer 特征处理
```

## 快速开始

```bash
pip install -r requirements.txt

# 预处理数据（FASTA → npy）
python AC/data.py

# 训练
python AC/train.py

# 预测
python AC/predict.py
```

## 依赖

PyTorch、scikit-learn、NumPy、Biopython，详见 `requirements.txt`。

## 说明

数据集（FASTA 序列文件，体积较大）与训练好的权重（`.pth`）不包含在仓库中，已在 `.gitignore` 中排除，本地保留。
