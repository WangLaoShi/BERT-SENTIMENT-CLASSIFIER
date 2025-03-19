import csv
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm

"""
读取评论文件的评论信息
"""
def read_file(file_name):
    comments_data = None
    # 读取评论信息
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        comments_data = [[line[0], int(line[1])] for line in reader]                      # 读取评论数据和对应的标签信息
    
    # 打乱数据集
    random.shuffle(comments_data)
    data = pd.DataFrame(comments_data)
    same_sentence_num = data.duplicated().sum()                                           # 统计重复的评论内容个数
    
    if same_sentence_num > 0:
        data = data.drop_duplicates()                                                     # 删除重复的样本信息
    
    f.close()
    
    return data


comments_data = read_file('./file/comments.csv')
print(len(comments_data))


# 划分训练集与测试集，并将pandas数据类型转化为列表类型
train_comments, train_labels = list(comments_data[: split_line][0]), list(comments_data[: split_line][1])
test_comments, test_labels = list(comments_data[split_line:][0]), list(comments_data[split_line:][1])

print(len(train_comments),len(train_labels), len(test_comments), len(test_labels))

from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
Step2: 定义BERTClassifier分类器模型
"""
class BERTClassifier(nn.Module):

    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self, output_dim, pretrained_name='bert-base-chinese'):

        super(BERTClassifier, self).__init__()
        
        # Step 2: 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_name,local_files_only=True)

        # 将模型移动到 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 定义 Bert 模型
        # self.bert = BertModel.from_pretrained(pretrained_name)
        self.bert = model

        # 外接全连接层
        self.mlp = nn.Linear(768, output_dim)


    def forward(self, tokens_X):

        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        res = self.bert(**tokens_X)

        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析 
        # return self.mlp(res[1])
#         📌 为什么会报错？
#         self.bert(**tokens_X) 返回的是一个 BaseModelOutputWithPoolingAndCrossAttentions 对象，不一定有 res[1]。
#         不同 BERT 变体，返回的 res 结构可能不同，正确的提取方式取决于 BERT 版本：

#         res[0]：最后一层隐藏状态（形状 [batch_size, seq_length, hidden_dim]）
#         res[1]（可能不存在）：pooler_output，即 [CLS] token 的嵌入（通常用于分类任务）
#         如果 res[1] 不存在，你应该用 res.pooler_output，或者 res[0][:, 0, :] 代替。
        # 直接使用 pooler_output 作为特征
        # return self.mlp(res.pooler_output)
        return res.logits



"""
评估函数，用以评估数据集在神经网络下的精确度
"""
def evaluate(net, comments_data, labels_data):
    
    sum_correct, i = 0, 0
    
    while i <= len(comments_data):
        
        comments = comments_data[i: min(i + 8, len(comments_data))]
        
        tokens_X = tokenizer(comments,
                             padding=True,
                             truncation=True,
                             max_length=512,  # 限制最大长度
                             return_tensors='pt').to(device=device)

        res = net(tokens_X)                                          # 获得到预测结果

        y = torch.tensor(labels_data[i: min(i + 8, len(comments_data))]).reshape(-1).to(device=device)

        sum_correct += (res.argmax(axis=1) == y).sum()              # 累加预测正确的结果
        i += 8

    return sum_correct/len(comments_data)                           # 返回(总正确结果/所有样本)，精确率


"""
训练bert_classifier分类器

"""
def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels, device, epochs):
    
    max_acc = 0.5                                 # 初始化模型最大精度为0.5
    
    # 累计训练 18 万条数据 epochs 次，优化模型
    for epoch in tqdm(range(epochs)):
        
        i, sum_loss = 0, 0                           # 每次开始训练时， i 为0 表示从第一条数据开始训练
        
        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels)
        test_acc = evaluate(net, test_comments, test_labels)
        
        # 输出精度
        print('\n--epoch', epoch, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc, '\t--test_acc', test_acc)
        
        
        
        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            
            # 更新历史最大精确度
            max_acc = test_acc
            
            # 保存模型
            torch.save(net.state_dict(), 'bert.parameters')
        
        
        # 开始训练模型
        while i < len(comments_data):
            comments = train_comments[i: min(i+8, len(train_comments))]             # 批量训练，每次训练8条样本数据

            # 通过 tokenizer 数据化输入的评论语句信息，准备输入bert分类器
            # 输入的8个评论语句长度很可能不一致，这时取长度为最长的那个句子，padding=True代表对短句子进行填充操作
            # 当输入的某个句子过长时，使用truncation=True进行截断操作
            # return_tensors='pt' 代表返回的数据类型为 python 的 torch 类型
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)

            # 将数据输入到bert分类器模型中，获得结果
            res = net(tokens_X)

            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i+8, len(train_comments))]).reshape(-1).to(device=device)

            optimizer.zero_grad()                  # 清空梯度
            l = loss(res, y)                       # 计算损失
            l.backward()                           # 后向传播
            optimizer.step()                      # 更新梯度

            sum_loss += l.detach()                # 累加损失
            i += 8                                # 样本下标累加


from d2l import torch as d2l

device = d2l.try_gpu()                                  # 获取GPU

net = BERTClassifier(output_dim=3)                      # BERTClassifier分类器，因为最终结果为3分类，所以输出维度为3，代表概率分布
net = net.to(device)


# 定义tokenizer对象，用于将评论语句转化为BertModel的输入信息
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

from modelscope.models import Model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese",local_files_only=True)

# Step 3: 移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("模型和 Tokenizer 加载完成！")


loss = nn.CrossEntropyLoss()                            # 损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)      # 小批量随机梯度下降算法


train_bert_classifier(net,
                      tokenizer,
                      loss,
                      optimizer,
                      train_comments,
                      train_labels,
                      test_comments,
                      test_labels,
                      device,
                      20)