import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 检查 CUDA 是否可用，如果可用则使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义序数值编码
def ordinal_encode_sequence(seq, max_len=1000):
    encoding = {'A': 0.25, 'T': 0.5, 'G': 0.75, 'C': 1.0, 'N': 0.0}
    seq_encoded = [encoding[base] for base in seq]
    seq_encoded += [0.0] * (max_len - len(seq_encoded))  # 填充为最大长度
    return seq_encoded

# 将序列数据转换为序数值编码
sequences_encoded = np.array([ordinal_encode_sequence(seq) for seq in sequences])

# 定义简化的 Transformer 模型
class SimplifiedTransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, num_layers=1, num_heads=4, dropout=0.5):
        super(SimplifiedTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        x = self.fc_out(x)
        return x

# 数据加载和处理
sequences = np.load('./numpy/sequences.npy')
labels = np.load('./numpy/labels.npy')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# 将 NumPy 数组转换为 PyTorch Tensor 并应用序数值编码
X_train_encoded = np.array([ordinal_encode_sequence(seq) for seq in X_train])
X_val_encoded = np.array([ordinal_encode_sequence(seq) for seq in X_val])

X_train = torch.tensor(X_train_encoded, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val_encoded, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = SimplifiedTransformerModel(input_size=1, hidden_size=64, output_size=len(np.unique(labels)))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

train(model, train_loader, val_loader, criterion, optimizer, epochs=10)

torch.save(model.state_dict(), './model.pth')
print("模型已保存到 './model.pth'")