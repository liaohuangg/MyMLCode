import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import random

input = torch.randn((3,5,10))
lstm1 = nn.LSTM(input_size=10, hidden_size=5, batch_first=True)

output1, (hn, cn) = lstm1(input)

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层，当设置 batch_first=True 时，会影响 output 的输出维度顺序
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化 h0 和 c0（隐藏状态和细胞状态）
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # LSTM 的向前传播，仅关注所有时间步的输出（output），忽略最终隐藏状态 (hn, cn)
        output, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))  

        # 输出层处理：对 LSTM 所有时间步的输出做线性变换
        out = self.fc(output[:, :, :])  
        return out