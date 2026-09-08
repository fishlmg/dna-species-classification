import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 定义最大序列长度和One-hot编码字典
max_len = 1000
one_hot_dict = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],
    'W': [0.5, 0.5, 0, 0],
    'S': [0, 0, 0.5, 0.5],
    'M': [0.5, 0, 0.5, 0],
    'K': [0, 0.5, 0, 0.5],
    'R': [0.5, 0, 0, 0.5],
    'Y': [0, 0.5, 0.5, 0],
    'B': [0, 0.3333, 0.3333, 0.3333],
    'D': [0.3333, 0, 0.3333, 0.3333],
    'H': [0.3333, 0.3333, 0, 0.3333],
    'V': [0.3333, 0.3333, 0.3333, 0],
    'X': [0, 0, 0, 0]
}

def one_hot_encode_sequence(seq):
    """将单条核苷酸序列转换为One-hot编码，并按最大长度填充或截断"""
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += 'N' * (max_len - len(seq))  # 填充为最大长度
    return [one_hot_dict.get(base, one_hot_dict['N']) for base in seq]

# 定义模型架构：CNN + LSTM/GRU + Transformer
class CNN_LSTM_GRU_Transformer(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=10, num_layers=2, num_heads=8, dropout=0.1):
        super(CNN_LSTM_GRU_Transformer, self).__init__()

        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM部分
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Transformer部分
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN部分
        x = x.transpose(1, 2)  # 转换形状: (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积
        
        # LSTM部分
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 只取LSTM的最后一个时间步的输出
        
        # Transformer部分
        x = self.transformer_encoder(x.unsqueeze(0))  # 添加批次维度
        x = x.squeeze(0)  # 去掉批次维度
        
        # 全连接层
        x = self.fc(x)  # 通过全连接层得到输出
        return x

# 数据加载和处理
sequences = np.load('./numpy/sequences.npy')
labels = np.load('./numpy/labels.npy')
species_names = np.load('./numpy/species_names.npy')

# 使用LabelEncoder将物种名称转换为标签
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(species_names)

# 转换为Tensor
sequences = torch.tensor(sequences, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据集并分割为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# 定义DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNN_LSTM_GRU_Transformer(input_size=4, hidden_size=128, output_size=len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

# 开始训练
train(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# 保存模型
torch.save(model.state_dict(), './model.pth')
print("模型已保存到 './model.pth'")

