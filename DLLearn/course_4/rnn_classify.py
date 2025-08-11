import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
# 检测是否有可用的CUDA设备，有则用'cuda'，否则用'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
数据预处理
'''
plt.rcParams['font.sans-serif'] = ['KaiTi']  #指定默认字体 SimHei黑体
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号'

df_train=pd.read_csv('/root/workspace/MyMLCode/DLLearn/dataset/IMDB/Train.csv')
df_test=pd.read_csv('/root/workspace/MyMLCode/DLLearn/dataset/IMDB/Test.csv')
df=pd.concat([df_train,df_test]).reset_index(drop='True')
# print("before", df_train.head())
# step1 应用预处理
def preprocess_text(s):
    # 解码字节字符串
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    # 替换转义字符
    s = re.sub(r"\\n", " ", s)
    s = re.sub(r"\\'", "'", s)
    return s.strip().replace("b'",'').replace('b"','')
df['text'] = df['text'].apply(preprocess_text)
# print(df_train['text'][0])
# print(df_train.shape)

# step2 分词
def get_tokenized_imdb(data):
    def tokenizer(text):
        # 替换<br />为空格
        text = re.sub(r"<br />", " ", text)
        # 保留字母和数字，其他替换为空格
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        # 按空格分割并转小写
        return [tok.lower() for tok in text.split(' ')]  # 加 if tok 过滤空字符串
    return [tokenizer(review) for review in data]

reviews = get_tokenized_imdb(df['text'])
# print(reviews[0])
# print(len(reviews))

# step3 提取不重复的单词
# 1. 提取所有单词并去重（集合自动去重）
# 先遍历reviews中的每一个review，再遍历每个review中的每个word，最后用set去重
all_words_set = set([word for review in reviews for word in review])

# 2. 构建 索引→单词 的映射（先转集合去重，再转列表）
idx_to_char = list(all_words_set)

# 3. 查看不重复单词总数
# print(len(idx_to_char))

# 4. 构建 单词→索引 的映射（通过枚举列表实现）
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
# print(char_to_idx['the'])  # 查看'the'的索引

# 5. 将每个review中的单词转换为索引
corpus_indices = []  
for i in range(len(reviews)):  
    indices = [char_to_idx[word] for word in reviews[i]]  
    corpus_indices.append(indices)  
# print("第一个文本中词的数量 ",len(corpus_indices[0]))  # 查看第一个review的索引

'''
embeding 编码
'''
# 将每个单词映射为一个十维的低维度向量
# 计算词汇表大小
vocab_size = len(idx_to_char)
# print("词汇表大小:", vocab_size)

# 定义嵌入维度和嵌入矩阵
embedding_dim = 10
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

# 定义嵌入函数，将索引序列转换为嵌入向量
def embedding(x):
    embedded = []
    for indices in x:
        embeddings = embedding_matrix[indices]
        embedded.append(torch.tensor(embeddings, dtype=torch.float32))
    return embedded

# 对第一条评论的索引序列做嵌入转换并查看结果
# first_review_embedding = embedding(corpus_indices[0])
# print("第一条评论嵌入结果:", first_review_embedding)

# 查看嵌入后张量的形状
embedding_shape = torch.tensor(embedding_matrix[corpus_indices[0]]).shape
# print("嵌入后张量形状:", embedding_shape)


'''
设备选择
'''
# 定义模型相关参数
num_inputs, num_hiddens, num_outputs = 10, 256, 2
# 打印将会使用的设备
print('will use', device)

'''
参数初始化
'''
def get_params():
    def _one(shape):
        # 生成符合正态分布的张量，转为 Parameter
        ts = torch.tensor(
            np.random.normal(0, 0.01, size=shape),
            device=device, 
            dtype=torch.float32
        )
        return torch.nn.Parameter(ts, requires_grad=True)
    
    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(
        torch.zeros(num_hiddens, device=device),
        requires_grad=True
    )
    
    # 输出层参数
    W_hy = _one((num_hiddens, num_outputs))
    b_y = torch.nn.Parameter(
        torch.zeros(num_outputs, device=device),
        requires_grad=True
    )
    
    # 用 ParameterList 管理所有参数
    return nn.ParameterList([W_xh, W_hh, b_h, W_hy, b_y])

# 权重偏置初始化
params = get_params()
# print("参数列表:", params)
# 隐藏状态初始化， 每一层有个隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    """初始化 RNN 的隐藏状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 初始化隐藏状态（batch_size=1）
state = init_rnn_state(1, num_hiddens, device)
# print("初始化的 RNN 隐藏状态:", state)

def rnn(inputs, state, params):
    """手动实现 RNN 前向传播逻辑"""
    W_xh, W_hh, b_h, W_hy, b_y = params
    H, = state  # 解包隐藏状态
    
    for X in inputs:
        # 计算隐藏层状态：H_t = tanh(W_xh @ X_t + W_hh @ H_{t-1} + b_h)
        H = torch.tanh(
            torch.matmul(X, W_xh) + 
            torch.matmul(H, W_hh) + 
            b_h
        )
        # 计算输出：Y_t = W_hy @ H_t + b_y
        Y = torch.matmul(H, W_hy) + b_y
    
    return Y, (H,)

# 执行 RNN 前向传播
inputs = [torch.randn( 1, 10,  device=device)]  # shape: (batch_size=1, 265)
Y, state_new = rnn(inputs, state, params)
print("输出 Y 的形状:", Y.shape)
# print("输出 Y :", Y)
# print("新的隐藏状态:", state_new)
'''
参数列表: ParameterList(
    (0): Parameter containing: [torch.float32 of size 10x256 (cuda:0)]
    (1): Parameter containing: [torch.float32 of size 256x256 (cuda:0)]
    (2): Parameter containing: [torch.float32 of size 256 (cuda:0)]
    (3): Parameter containing: [torch.float32 of size 256x2 (cuda:0)]
    (4): Parameter containing: [torch.float32 of size 2 (cuda:0)]
)
'''
'''
预测结果
'''
probabilities = torch.softmax(Y, dim=1)
print("预测概率:", probabilities)
sentiment_map = {1:"Positive", 0:"Negative"}
int(Y.argmax())
print("预测情感类别:", sentiment_map[int(Y.argmax())])

def sentiment_analysis(data):
    # 1. 单词→索引映射
    index_data = [char_to_idx[word] for word in data]
    # 2. 索引→嵌入向量
    input_data = embedding([index_data])[0]  # shape: (seq_len, embedding_dim)
    # 3. 拆成列表，每个元素 shape 为 (1, embedding_dim)，并放到 device 上
    inputs = [x.unsqueeze(0).to(device) for x in input_data]
    # 4. 初始化隐藏状态
    state = init_rnn_state(1, num_hiddens, device)
    # 5. RNN 前向传播
    Y, state_new = rnn(inputs, state, params)
    # 6. 预测情感（索引→文本映射）
    return sentiment_map[int(Y.argmax())]

# ------------------- 执行预测 -------------------
# 对第 845 条评论（模拟）执行情感分析
result = sentiment_analysis(reviews[845])  # 这里用 reviews[0] 模拟，需替换为真实数据
print("预测情感:", result)