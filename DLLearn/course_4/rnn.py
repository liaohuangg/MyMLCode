import torch
import torch.nn as nn

class myRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=4, output_size=3):
        super(myRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 创建一个RNN模块，里面自然就包含了4个隐藏层
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)

        # 输出层是需要单独建立的
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x的形状: (seq_length, batch_size, input_size)
        # 初始化h0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # 传入x
        output, hn = self.rnn(x, h0)

        # 我们关心的是最后一个时间步的输出，可以从output中索引
        predict = self.fc(output[-1, :, :])
        
        # 也可以从hn中索引
        # predict = self.fc(hn[-1,:,:])
        return predict


class myRNN4(nn.Module):
    def __init__(self, input_size=100, hidden_sizes=[256, 256, 512, 512], output_size=3):
        super(myRNN4, self).__init__()
        # 定义4个不同的rnn层
        self.rnn1 = nn.RNN(input_size, hidden_sizes[0])
        self.rnn2 = nn.RNN(hidden_sizes[0], hidden_sizes[1])
        self.rnn3 = nn.RNN(hidden_sizes[1], hidden_sizes[2])
        self.rnn4 = nn.RNN(hidden_sizes[2], hidden_sizes[3])
        # 定义输出层
        self.linear = nn.Linear(hidden_sizes[3], output_size)

    def forward(self, x):
        # 初始化h0，对4个隐藏层分别初始化
        # hidden_sizes 需在类中有定义或传入，这里假设可通过 self 访问
        h0 = [
            torch.zeros(1, x.size(0), hidden_sizes[0]).to(x.device),
            torch.zeros(1, x.size(0), hidden_sizes[1]).to(x.device),
            torch.zeros(1, x.size(0), hidden_sizes[2]).to(x.device),
            torch.zeros(1, x.size(0), hidden_sizes[3]).to(x.device)
        ]
        
        # 手动循环传递，让输出进入下一个RNN层
        output1, _ = self.rnn1(x, h0[0])
        output2, _ = self.rnn2(output1, h0[1])
        output3, _ = self.rnn3(output2, h0[2])
        output4, _ = self.rnn4(output3, h0[3])
        
        # 取出最后一个时间步的结果，通过输出层
        output = self.linear(output4[-1, :, :])
        return output

models = myRNN4()
print(models)

# 生成随机输入张量，形状为(seq_len, batch_size, input_size)
inputs4 = torch.randn((3, 50, 10))  # seq_len (vocal_size), batch_size, input_size
# 定义一个双向 RNN 模型，input_size=10, hidden_size=20, 双向
brnn = nn.RNN(input_size=10, hidden_size=20, bidirectional=True)