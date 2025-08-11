import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import random

'''
数据预处理
'''
with open('/root/workspace/MyMLCode/DLLearn/dataset/JAYCHOU/jaychou_lyrics.txt', 'r', encoding='utf-8') as f:
    corpus_char = f.read()
corpus_char = corpus_char.replace('\n', ' ')
corpus_chars = corpus_char[:15000]
# set 去重，list 保持顺序, 构建索引到字符的映射
idx_to_char = list(set(corpus_chars)) 
# 构建字符到索引的映射
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

# 将原本的文本数据转化为索引表示
corpus_indices = [char_to_idx[char] for char in corpus_chars]

'''
数据预处理
'''
# 语料库比较小，可以采用独热编码
def one_hot(x, n_class, dtype=torch.float32):
    """手动实现 one-hot 编码"""
    # 转换为长整型（避免索引类型错误）
    x = x.long()
    
    # 初始化全 0 张量，形状: [x.shape[0], n_class]
    res = torch.zeros(
        x.shape[0],    # 样本数量（x 的长度）
        n_class,       # 类别总数
        dtype=dtype,   # 数据类型
        device=x.device  # 设备（与输入 x 一致）
    )
    
    # scatter_ 实现 one-hot：在维度 1 上，按 x 的索引填充 1
    res.scatter_(1, x.view(-1, 1), 1)
    return res

# ------------------- 示例验证 -------------------
# 模拟输入（类别索引）
#x = torch.tensor([0, 2])

# 假设词汇表大小（类别总数）
#vocab_size = 3  # 需与实际场景一致，这里用之前的 vocab_size 示例值

# 执行 one-hot 编码
# one_hot_result = one_hot(x, vocab_size)
# print("one-hot 编码结果:\n", one_hot_result)
# print("编码形状:", one_hot_result.shape)
vocab_size = len(idx_to_char)
one_hot_corpus = one_hot(torch.tensor(corpus_indices), n_class=vocab_size)
# print("one-hot 编码形状:", one_hot_corpus.shape)

'''
网络结构和设备选择
'''
num_inputs, num_hiddens, num_outputs = len(idx_to_char), 256, vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)
print("输入维度:", num_inputs, "隐藏层维度:", num_hiddens, "输出维度:", num_outputs)

'''
权重偏置初始化
'''
def get_params():
    def normal(shape):
        return torch.randn(shape, device=device) * 0.01
    
    def three_layer_init():
        return (normal((num_inputs, num_hiddens)), 
                normal((num_hiddens, num_hiddens)), 
                torch.zeros(num_hiddens, device=device))
    
    # 隐藏层参数
    W_xh, W_hh, b_h = three_layer_init()

    # 输出层参数
    W_hy = normal((num_hiddens, num_outputs))
    b_y = torch.zeros(num_outputs, device=device)
    
    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.requires_grad_(True)
    
    return params

params = get_params()

'''
隐藏状态初始化
'''
def init_rnn_state(batch_size, num_hiddens, device):
    """初始化 RNN 隐藏状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device))
state = init_rnn_state(batch_size=1, num_hiddens=num_hiddens, device=device)
# print(state)

'''
RNN 前向计算
'''
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

'''
预测函数
'''
def to_onehot(X, n_class):
    # X: shape (batch_size, seq_len)
    # 返回: 长度为 seq_len 的列表，每个元素 shape (batch_size, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, 
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    """用 RNN 预测序列（字符级）"""
    # 初始化 RNN 隐藏状态（batch_size=1）
    state = init_rnn_state(1, num_hiddens, device)
    
    # 初始输出：prefix 的第一个字符索引
    output = [char_to_idx[prefix[0]]]
    
    # 循环生成字符（总长度 = num_chars + len(prefix) - 1）
    for t in range(num_chars + len(prefix) - 1):
        # 当前输入：上一时刻输出的 one-hot 编码
        X = to_onehot(
            torch.tensor([[output[-1]]], device=device), 
            vocab_size
        )
        
        # RNN 前向传播，更新隐藏状态
        (Y, state) = rnn(X, state, params)
        
        # 决定下一时刻输入：prefix 字符 或 预测字符
        if t < len(prefix) - 1:
            # 前 len(prefix)-1 步用 prefix 的真实字符
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # 后续用预测字符（取概率最大的索引）
            output.append(int(Y[0].argmax().item()))
    
    # 索引转字符，拼接成结果
    return ''.join([idx_to_char[i % len(idx_to_char)] for i in output])

# ------------------- 执行预测 -------------------
# 预测以 '分开' 为前缀，生成 10 个字符
result = predict_rnn(
    '分开', 10, rnn, params, init_rnn_state, 
    num_hiddens, vocab_size, device, idx_to_char, char_to_idx
)
print("预测序列:", result)