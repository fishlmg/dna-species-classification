import torch
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# 定义最大序列长度和One-hot编码字典
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

# 加载模型和物种名称映射
model = CNN_LSTM_GRU_Transformer(input_size=4, hidden_size=128, output_size=10)  # 临时output_size
model.load_state_dict(torch.load('./model.pth'))
model.eval()

# 使用LabelEncoder将物种名称转换为标签
species_names = np.load('./numpy/species_names.npy')
label_encoder = LabelEncoder()
label_encoder.fit(species_names)

# 更新模型输出尺寸
output_size = len(label_encoder.classes_)
model.fc = nn.Linear(model.fc.in_features, output_size)  # 根据物种数目更新输出层

# 预测函数
def predict(model, fasta_file):
    encoded_sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        encoded_seq = one_hot_encode_sequence(seq)
        encoded_sequences.append(encoded_seq)
    
    seq_tensor = torch.tensor(encoded_sequences, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(seq_tensor)
        _, predicted = torch.max(outputs, 1)
    
    species_pred = label_encoder.inverse_transform(predicted.numpy())
    return species_pred

# 主程序
def main():
    test_file = './test_sequence.fasta'  # 输入测试FASTA文件路径
    predictions = predict(model, test_file)
    print(f"预测物种名称：{predictions}")

if __name__ == '__main__':
    main()
