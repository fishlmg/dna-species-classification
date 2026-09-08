import os
import numpy as np
from Bio import SeqIO

# 定义最大序列长度和扩展后的One-hot编码字典，添加更多IUPAC字符支持
max_len = 1000
one_hot_dict = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],  # 用于填充和任意碱基
    'W': [0.5, 0.5, 0, 0],  # A或T
    'S': [0, 0, 0.5, 0.5],  # C或G
    'M': [0.5, 0, 0.5, 0],  # A或C
    'K': [0, 0.5, 0, 0.5],  # G或T
    'R': [0.5, 0, 0, 0.5],  # A或G
    'Y': [0, 0.5, 0.5, 0],  # C或T
    'B': [0, 0.3333, 0.3333, 0.3333],  # C、G或T
    'D': [0.3333, 0, 0.3333, 0.3333],  # A、G或T
    'H': [0.3333, 0.3333, 0, 0.3333],  # A、C或T
    'V': [0.3333, 0.3333, 0.3333, 0],  # A、C或G
    'X': [0, 0, 0, 0]  # 无法确定的碱基，统一归为N
}

def one_hot_encode_sequence(seq):
    """将单条核苷酸序列转换为One-hot编码，并按最大长度填充或截断"""
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += 'N' * (max_len - len(seq))  # 填充为最大长度
    return [one_hot_dict.get(base, one_hot_dict['N']) for base in seq]  # 使用one_hot_dict.get来处理未知字符

def load_fasta_data_from_folder(folder_path):
    """从文件夹中读取所有FASTA文件并进行One-hot编码"""
    sequences = []
    labels = []
    species_names = []
    label_dict = {}

    # 遍历文件夹中的所有FASTA文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 只处理FASTA文件
        if file_path.endswith(".fasta") or file_path.endswith(".fa"):
            for record in SeqIO.parse(file_path, "fasta"):
                # 提取物种名称（前四个部分作为物种名称），并去掉末尾的逗号或其他符号
                description_parts = record.description.split()
                if len(description_parts) > 4:
                    species_name = " ".join(description_parts[:4]).strip(",")
                else:
                    species_name = " ".join(description_parts).strip(",")
                
                # 如果物种名称未被记录，则分配新标签
                if species_name not in label_dict:
                    label_dict[species_name] = len(label_dict)
                    species_names.append(species_name)
                label = label_dict[species_name]
                
                # One-hot编码并添加到列表中
                seq = str(record.seq)
                encoded_seq = one_hot_encode_sequence(seq)
                sequences.append(encoded_seq)
                labels.append(label)

    # 确保 sequences 和 labels 数组长度一致
    if len(sequences) != len(labels):
        raise ValueError(f"Mismatch between sequences ({len(sequences)}) and labels ({len(labels)}) lengths!")
    
    # 转换为NumPy数组
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    return sequences, labels, species_names

# 加载数据
folder_path = "./data"  # 请替换为您的文件夹路径
sequences, labels, species_names = load_fasta_data_from_folder(folder_path)

# 指定保存路径
save_folder = "./numpy"  # 请替换为目标文件夹路径
if not os.path.exists(save_folder):
    os.makedirs(save_folder)  # 如果目标文件夹不存在，创建它

# 保存数据到指定文件夹
np.save(os.path.join(save_folder, "sequences.npy"), sequences)
np.save(os.path.join(save_folder, "labels.npy"), labels)
np.save(os.path.join(save_folder, "species_names.npy"), np.array(species_names))

print(f"数据已保存到 {save_folder} 文件夹中：")
print(f"sequences.npy, labels.npy 和 species_names.npy")
