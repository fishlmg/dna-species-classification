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
- 从登录号清单经 NCBI Entrez 拉取序列片段
- 完整的训练 → 测试 → 预测流程
- NumPy 预处理缓存（`.npy`），避免重复解析 FASTA

## 🗂 项目结构

```
.
├── AC/                          # One-hot + 神经网络方案
│   ├── download_genbank.py      # 按登录号用 Entrez 下载 FASTA 片段
│   ├── data.py                  # FASTA 读取、编码、数据集生成
│   ├── data_test.py             # 数据管道测试
│   ├── train.py                 # CNN + LSTM + Transformer 训练
│   ├── model_test.py            # 另一套 CNN 的全量打分
│   ├── test.py / test1.py       # LSTM 训练时打印验证准确率 / 序数编码试验
│   ├── predict.py               # 单序列推理预测
│   ├── AC.txt / ac1~3.txt       # GenBank / RefSeq 登录号清单
│   ├── data/                    # FASTA 原始序列（不入库）
│   └── numpy/                   # 预处理后的 .npy 缓存（不入库）
└── k_mer/
    └── data.py                  # k-mer 特征提取（自带 3 条示例序列）
```

## 🚀 快速开始

`data.py`、`train.py`、`test.py` 都使用当前目录下的 `./data` 和 `./numpy`，请在 `AC/` 里运行。

```bash
pip install -r requirements.txt
cd AC

# 只核对登录号，不访问 NCBI
python download_genbank.py --dry-run

# 拉取 FASTA。邮箱必填；API key 可选，用来提高 NCBI 速率限额
export NCBI_EMAIL="you@example.com"
python download_genbank.py --limit 5     # 先试 5 条
python download_genbank.py               # 清单中的全部去重登录号

python data.py     # FASTA → numpy/*.npy
python test.py     # 训练 LSTM，终端打印每一轮的 Validation Accuracy
```

## 📥 数据获取

仓库不包含 FASTA。序列来自 NCBI Nucleotide（GenBank / RefSeq），用登录号经 Entrez 取出，再交给 `data.py`。

四份清单合计 **612** 个去重登录号（`AC_*` 93 个，`NC_*` 519 个）：

| 文件 | 格式 | 内容 |
| --- | --- | --- |
| `AC/AC.txt` | 每行一个 | `AC_000008.1`–`AC_000100.1`，共 93 条 |
| `AC/ac1.txt` | 每行一个 | 59 条 `AC_*`，全部包含在 `AC.txt` 中；`AC.txt` 另有 34 条 |
| `AC/ac2.txt` | 每行一个 | `NC_*`，文件中有重复，去重后 250 条 |
| `AC/ac3.txt` | 逗号分隔 | `NC_*`，去重后 269 条。其中一处是全角逗号，`NC_001815` 的版本号被截断 |

这些登录号混有短记录和整条染色体，不能按“物种条形码小文件”来估体积。抽查过的例子：

| 登录号 | 记录 | 长度 |
| --- | --- | --- |
| `AC_000008.1` | Human adenovirus 5, complete genome | 35,938 bp |
| `NC_002084.1` | Anopheles gambiae mitochondrion | 15,363 bp |
| `NC_002745.2` | Staphylococcus aureus subsp. aureus N315 | 2,814,816 bp |
| `NC_000913.3` | Escherichia coli str. K-12 substr. MG1655 | 4,641,652 bp |
| `AC_000048.1` | Homo sapiens chromosome 5, alternate assembly | 176,344,326 bp |
| `NC_000001.11` | Homo sapiens chromosome 1, GRCh38.p14 | 248,956,422 bp |

`data.py` 的 `max_len = 1000`：长于 1000 bp 的序列只保留开头，短的用 `N` 补齐。`download_genbank.py` 因此默认向 Entrez 要 `seq_start=1&seq_stop=1000` 的 FASTA 窗口，而不是整条染色体。需要更长窗口时用 `--seq-stop`；编码阶段仍会截到 1000 bp。

手动取一条的等价请求（[E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)）：

```text
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=AC_000008.1&rettype=fasta&retmode=text&seq_start=1&seq_stop=1000
```

脚本对应的是 Biopython `Bio.Entrez.efetch(db="nuccore", rettype="fasta", retmode="text", seq_start=1, seq_stop=1000)`。NCBI 要求带联系邮箱，无 API key 时把速率放在约每秒 3 次以内；脚本在两次请求之间睡眠，失败的登录号跳过并继续。FASTA 写到 `AC/data/<登录号>.fasta`，该目录已在 `.gitignore` 中。

`data.py` 的类别名取自 FASTA 标题按空白切开后的前四个字段，并去掉末尾逗号。带坐标窗口的标题形如 `AC_000008.1:1-1000 Human adenovirus 5, complete genome`，类别名会是 `AC_000008.1:1-1000 Human adenovirus 5`。同一物种的不同登录号会变成不同类别。

`k_mer/data.py` 不读这些清单。它用文件底部写死的 3 条短序列演示 k-mer 计数和 L2 归一化。

## 📊 结果

仓库里没有训练日志、权重或评测数字，下表的数值保持为「未记录」。在 `AC/` 下生成 `numpy/sequences.npy` 与 `numpy/labels.npy` 之后，用对应命令跑一遍，把终端最后一轮打印的数字填回表中。

| 运行 | 模型 | 划分 | 看终端里的哪一行 | 数值 |
| --- | --- | --- | --- | --- |
| `python test.py` | LSTM（`input_size=4, hidden_size=128`，Adam `lr=0.001`，10 epoch） | `train_test_split(test_size=0.2, random_state=42)`，标签来自 `labels.npy` | 最后一轮 `Validation Accuracy` | 未记录 |
| `python train.py` | CNN + LSTM + Transformer，同样的学习率和 epoch | 同样的 `test_size=0.2, random_state=42` | 最后一轮 `Validation Accuracy` | 未记录 |
| `python model_test.py` | `SpeciesClassifier`，权重文件 `best_species_classifier.pth` | 对全部缓存序列打分 | `测试集准确率` | 未记录 |

读数时对应脚本里的实现：

- `test.py` 每次都从头训练，并用这一轮的验证集准确率覆盖打印；同时把权重写到 `model.pth`。打印的是这次运行的验证集结果。
- `train.py` 先加载 `labels.npy`，再用 `species_names.npy` 重新做 `LabelEncoder`。`data.py` 写入的 `species_names.npy` 是类别名列表，长度等于类别数；`sequences.npy` 的长度是序列条数。两者不一致时，`train_test_split` 会在打印准确率之前报错。按序列对齐的标签在 `labels.npy`，`test.py` 用的是这个文件。
- `model_test.py` 读取 `best_species_classifier.pth`。`train.py` 写出的文件名是 `model.pth`，两者也不是同一个网络。`model_test.py` 统计的是全部缓存序列。
- `data_test.py` 只打印数组形状和物种名。`k_mer/data.py` 没有分类准确率。

## 🧬 编码说明

One-hot 编码字典（含简并碱基）：

| 碱基 | 编码 | 含义 |
| --- | --- | --- |
| A / T / C / G | `[1,0,0,0]` 等 | 标准碱基 |
| N | `[0,0,0,0]` | 未知碱基 |
| W / S / M / K | 两个碱基各 0.5 | 简并碱基（如 W = A/T） |

## 📦 依赖

PyTorch、scikit-learn、NumPy、Biopython —— 详见 [requirements.txt](requirements.txt)

Entrez 下载用的是 Biopython 自带的 `Bio.Entrez`，不需要额外包。联系邮箱和可选的 API key 只放在环境变量 `NCBI_EMAIL`、`NCBI_API_KEY` 里。

## 📝 说明

- FASTA、`.npy` 缓存和训练权重（`.pth`）不包含在仓库中，已通过 `.gitignore` 排除
- 代码以学习/实验为目的，模型结构与超参数可按需调整
- 准确率以你本机运行 `AC/test.py` 打印的验证集数字为准；仓库不预填这个数

## 📄 许可证

[MIT License](LICENSE)
