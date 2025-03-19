import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
import csv
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 1. 读取数据并预处理
# ============================
class SentimentDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length=512):
        self.data = self.read_file(file_name)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def read_file(self, file_name):
        with open(file_name, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            data = [[line[0], int(line[1])] for line in reader if len(line) >= 2 and len(line[0]) > 0]
        random.shuffle(data)  # 打乱数据
        return pd.DataFrame(data, columns=["text", "label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================
# 2. 定义 BERT 分类模型
# ============================
class BERTSentimentClassifier(nn.Module):
    def __init__(self, pretrained_name='bert-base-chinese', num_classes=3):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 位置的向量
        output = self.dropout(cls_embedding)
        return self.fc(output)

# ============================
# 3. 训练函数
# ============================
def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs=5):
    model.to(device)
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练中")

        for batch in progress_bar:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"Loss": f"{total_loss / total:.4f}", "Acc": f"{correct / total:.4f}"})

        # 计算测试集准确率
        test_acc = evaluate_model(model, test_loader)
        print(f"🔍 Epoch {epoch+1} 测试集准确率: {test_acc:.4f}")

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_bert_sentiment_model.pth")
            print(f"✅ 最优模型已保存！Test Acc: {best_acc:.4f}")

# ============================
# 4. 评估函数
# ============================
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ============================
# 5. 运行训练和评估
# ============================
if __name__ == "__main__":
    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 读取数据
    dataset = SentimentDataset("./file/comments.csv", tokenizer)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型
    model = BERTSentimentClassifier()

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # 训练模型
    train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs=5)

    # 测试最佳模型
    model.load_state_dict(torch.load("best_bert_sentiment_model.pth"))
    final_acc = evaluate_model(model, test_loader)
    print(f"🏆 最终测试集准确率: {final_acc:.4f}")
