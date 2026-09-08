import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
# 定义模型类
class SpeciesClassifier(nn.Module):
    def __init__(self, num_classes, sequence_length):
        super(SpeciesClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64 * (sequence_length // 2), num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将 (batch_size, seq_len, 4) 转换为 (batch_size, 4, seq_len)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 加载数据
sequences = np.load(r"./numpy/sequences.npy")  # 请替换为您的文件路径
labels = np.load(r"./numpy/labels.npy")
species_names = np.load(r"./numpy/species_names.npy")


# 转换为 PyTorch 张量
X = torch.tensor(sequences, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# 创建 DataLoader
test_dataset = TensorDataset(X, y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载训练好的模型
model = SpeciesClassifier(num_classes=len(species_names), sequence_length=1000)
model.load_state_dict(torch.load(r"best_species_classifier.pth"))
model.eval()  # 切换到评估模式

# 进行推理
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")
