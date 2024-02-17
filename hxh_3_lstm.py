import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 假设每个字是1x13维的向量
input_dim = 13
hidden_dim = 64
output_dim = 2  # 两类情感，你可以根据实际情况调整

# 定义模型
class TextSentimentModel(nn.Module):
    def __init__(self):
        super(TextSentimentModel, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        print(text.shape)
        print(text)
        output, _ = self.rnn(text)
        last_hidden_state = output[:, -1, :]
        sentiment_logits = self.fc(last_hidden_state)
        return sentiment_logits

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_sequence = self.texts[idx]
        text_tensor = torch.FloatTensor([item.numpy() for item in text_sequence])
        label_item = self.labels[idx].item() if torch.is_tensor(self.labels[idx]) else self.labels[idx]
        return text_tensor, label_item


# 数据预处理
# 建立文本和标签的数据集，确保每个字都是1x13维的向量
# 示例文本序列
text_sequence_1 = [
    torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    torch.FloatTensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
    torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    torch.FloatTensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
]

text_sequence_2 = [
    torch.FloatTensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
    torch.FloatTensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
    torch.FloatTensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
    torch.FloatTensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
]

# 转换为 PyTorch FloatTensor
your_texts = [text_sequence_1, text_sequence_2]

# 示例标签
labels = [0, 1]

# 转换为 PyTorch Tensor
your_labels = torch.tensor(labels)

# 模型、数据集和优化器的初始化
model = TextSentimentModel()
dataset = TextDataset(your_texts, your_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    for batch_text, batch_label in dataloader:
        optimizer.zero_grad()
        output = model(batch_text)
        loss = criterion(output, batch_label)
        loss.backward()
        optimizer.step()

# 模型评估
# 在另外的测试集上进行评估，计算准确率等指标
