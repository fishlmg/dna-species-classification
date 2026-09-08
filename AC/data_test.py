import numpy as np

# 加载数据
sequences = np.load("./numpy/sequences.npy")
labels = np.load("./numpy/labels.npy")
species_names = np.load("./numpy/species_names.npy")

print("数据加载成功！")
print(f"Sequences length: {len(sequences)}")
print(f"Labels length: {len(labels)}")
print("序列数据形状:", sequences.shape)
print("标签数据形状:", labels.shape)
print("物种名称:", species_names)
