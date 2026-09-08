import numpy as np
import itertools
from sklearn.preprocessing import normalize
import torch
import os

# 生成k-mer字典
def generate_kmer_dict(k=3):
    """生成所有可能的k-mer字典"""
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=k)]
    kmer_dict = {kmer: i for i, kmer in enumerate(kmers)}
    return kmer_dict

# 计算DNA序列的k-mer计数向量
def kmer_count_vector(sequence, k=3, kmer_dict=None):
    """计算DNA序列的k-mer计数向量"""
    kmer_count = np.zeros(len(kmer_dict))  # 初始化计数向量
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_dict:
            kmer_count[kmer_dict[kmer]] += 1
    return kmer_count

# 处理一组DNA序列，提取k-mer特征向量
def process_sequences(sequences, k=3, kmer_dict=None):
    """处理一组DNA序列，提取k-mer特征向量"""
    feature_vectors = []
    for seq in sequences:
        # 去除非碱基字符（如N，可能会出现在一些序列中）
        seq = ''.join([base for base in seq if base in kmer_dict])
        if len(seq) >= k:  # 确保序列长度足够
            feature_vectors.append(kmer_count_vector(seq, k, kmer_dict))
    return np.array(feature_vectors)

# 生成序列的标签（与每个k-mer向量对应）
def generate_labels(sequences, labels, k=3):
    """生成序列的标签（与每个k-mer向量对应）"""
    label_vectors = []
    for seq, label in zip(sequences, labels):
        # 为每个序列生成标签向量，长度与序列相同
        label_vector = np.zeros(len(seq))
        for feature in label:
            start, end, feature_type = feature
            label_vector[start:end+1] = feature_type  # 假设特征类型是数字编码（如1=启动子，2=终止子，0=非功能区）
        label_vectors.append(label_vector)
    return np.array(label_vectors)

# 对特征向量进行标准化（L2 归一化）
def normalize_features(features):
    """对特征向量进行L2归一化"""
    return normalize(features, norm='l2')

# 保存处理后的数据
def save_data(features, labels, feature_filename="kmer_features.npy", label_filename="kmer_labels.npy"):
    """保存特征和标签到文件"""
    np.save(feature_filename, features)
    np.save(label_filename, labels)
    print(f"Features saved to {feature_filename}")
    print(f"Labels saved to {label_filename}")

# 主程序：处理数据并保存
def main(sequences, labels, k=3):
    # 生成 k-mer 字典
    kmer_dict = generate_kmer_dict(k)
    
    # 提取特征（k-mer 计数向量）
    feature_vectors = process_sequences(sequences, k, kmer_dict)
    
    # 对特征进行标准化
    feature_vectors_normalized = normalize_features(feature_vectors)
    
    # 生成标签（假设标签是区间和类型的元组）
    label_vectors = generate_labels(sequences, labels, k)
    
    # 保存处理后的数据
    save_data(feature_vectors_normalized, label_vectors)

# 示例序列和标签
sequences = [
    "AGCTAGCTAGC",  # 示例DNA序列1
    "GCTAGCTAGCTA",  # 示例DNA序列2
    "TGCATGCA"  # 示例DNA序列3
]

# 示例标签：每条序列的功能位点，格式为[(start, end, feature_type)]，其中feature_type为数字编码
labels = [
    [(0, 3, 1), (8, 11, 2)],  # 序列1: 启动子在(0, 3)，终止子在(8, 11)
    [(4, 7, 1), (10, 12, 2)], # 序列2: 启动子在(4, 7)，终止子在(10, 12)
    [(2, 5, 1)]               # 序列3: 启动子在(2, 5)
]

# 执行主程序，k=3
main(sequences, labels, k=3)
