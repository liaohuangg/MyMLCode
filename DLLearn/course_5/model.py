import torch
import torch.nn as nn
import math
# class MultiHeadAttention(nn.Module):


class Positional_Encoding(nn.Module):
    def __init__(self, embedding_size, seq_len, device):
        # 这个max_len是指句子中最大单词数
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(seq_len, embedding_size, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2, device=device).float() / embedding_size * -(math.log(10000.0)))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        '''
        使形状(seq_len, embedding_size) -> (1, seq_len, embedding_size)
        该张量会成为 nn.Module 模块的一个持久化属性，当使用 torch.save 保存模型时，
        缓冲区会和模型的参数（nn.Parameter ）一起被保存；
        加载模型时，也会被正确恢复。
        它不会被当作模型的可训练参数（即不会出现在 model.parameters() 迭代器中 ），
        因为位置编码是按照固定公式计算好的，不需要在训练过程中更新 ，避免了不必要的参数更新开销。
        '''
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区s

    def forward(self, x):
        # 将位置编码添加到输入中
        # (batch, seq_len, embedding_size), (1, seq_len, embedding_size)
        return x + self.pe[:, :x.size(1)] # 第一维度取seq_len个位置，保证相加的准确

class Multi_Head_Atten(nn.Module):
    def __init__(self, embedding_size, num_heads, device):
        super(Multi_Head_Atten, self).__init__()
        assert embedding_size % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = embedding_size
        self.num_heads = num_heads
        self.d_k = embedding_size // num_heads # 每个头的维度
        self.device = device
        #定义线性变换层 Q @ W_Q
        '''
        线性层：
        torch.nn.Linear(in_features, # 输入的神经元个数
           out_features, # 输出神经元个数
           bias=True # 是否包含偏置
           )
        卷积层：
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        输入是带通道的空间数据（(batch, in_channels, H, W)），
        输出是提取到的多通道特征图（(batch, out_channels, out_H, out_W)）
        '''
        self.W_q = nn.Linear(self.d_model, self.d_model).to(device)
        self.W_k = nn.Linear(self.d_model, self.d_model).to(device)
        self.W_v = nn.Linear(self.d_model, self.d_model).to(device)
        self.W_o = nn.Linear(self.d_model, self.d_model).to(device)

    def split_heads(self, x):
        """
        将输入张量分割为多个头
        输入形状: (batch_size, seq_length, d_model)
        输出形状: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        输入形状：
            Q: (batch_size, num_heads, seq_length, d_k) 从split_heads的输出来的
            K, V: 同Q
        输出形状： (batch_size, num_heads, seq_length, d_k)
        """
        # 计算注意力分数（Q和K的点积）Q * K.T, 交换倒数1 2 维度
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 输出的形状：  (batch_size, num_heads, seq_length, seq_length)
    
        # 应用掩码（如填充掩码或未来信息掩码）， 把0位置，也就是pad位置，用一个极小值代替
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    
        # 计算注意力权重（softmax归一化）
        attn_probs = torch.softmax(attn_scores, dim=-1)
    
        # 对值向量加权求和
        # 输入的形状：  (batch_size, num_heads, seq_length, seq_length)
        # 输出的形状：  (batch_size, num_heads, seq_length, d_k)
        output = torch.matmul(attn_probs, V)
        return output

    def combine_heads(self, x):
        """
        将输入张量的多个头合并会原始形状
        输入形状: (batch_size, num_heads, seq_length, d_k)
        输出形状: (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_len, d_k = x.size()
        # .contiguous() 保证张量在内存中是连续存储的
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


    def forward(self, src_Q,  src_K,  src_V, src_mask = None):
        # 输入Q K W, Mask 
        """
        前向传播
        输入形状: Q/K/V: (batch_size, seq_length, d_model)
        输出形状: (batch_size, seq_length, d_model)
        """
        # 线性变换并分割多头
        Q = self.split_heads(self.W_q(src_Q)) # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_k(src_K))
        V = self.split_heads(self.W_v(src_V))
       
        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, src_mask)
       
        # 合并多头并输出变换
        output = self.W_o(self.combine_heads(attn_output))
        return output

class Feedback_Forward(nn.Module):
    def __init__(self, embedding_size, d_ffn, device):
        super(Feedback_Forward, self).__init__()
        self.fc1_layer = nn.Linear(embedding_size, d_ffn).to(device)
        self.relu_layer = nn.ReLU()
        self.fc2_layer = nn.Linear(d_ffn, embedding_size).to(device)

    def forward(self, x):
        fc1 = self.fc1_layer(x)
        relu = self.relu_layer(fc1)
        fc2  = self.fc2_layer(relu)
        return fc2

class Encoder(nn.Module):
    def __init__(self, embedding_size, num_heads, d_ffn, dropout, device):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        # layer
        self.Multi_Atten_layer = Multi_Head_Atten(self.embedding_size, num_heads, device).to(device)
        self.FFN_layer = Feedback_Forward(self.embedding_size, d_ffn, device).to(device)
        self.norm1 = nn.LayerNorm(self.embedding_size).to(device)  # 层归一化
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)  # Dropout

    def forward(self, src, src_mask):
        # 自注意网络
        atten_output = self.Multi_Atten_layer(src, src, src, src_mask)
        x = self.norm1(src + self.dropout(atten_output))

        # 前馈网络
        ffn_output = self.FFN_layer(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_size, num_heads, d_ffn, dropout, device):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        
        # layer
        self.Multi_Atten_layer = Multi_Head_Atten(self.embedding_size, num_heads, device).to(device)
        self.Cross_Atten_layer = Multi_Head_Atten(self.embedding_size, num_heads, device).to(device)
        self.FFN_layer = Feedback_Forward(self.embedding_size, d_ffn, device).to(device)
        self.norm1 = nn.LayerNorm(self.embedding_size).to(device)
        self.norm2 = nn.LayerNorm(self.embedding_size).to(device)
        self.norm3 = nn.LayerNorm(self.embedding_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device) # Dropout

    def forward(self, encode_output, trg, trg_mask, src_mask):
        # 自注意网络
        atten_output = self.Multi_Atten_layer(trg, trg, trg, trg_mask)
        x = self.norm1(trg + self.dropout(atten_output))

        # 交叉注意网络
        cross_output = self.Cross_Atten_layer(x, encode_output, encode_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # 前馈网络
        ffn_output = self.FFN_layer(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
        

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 embedding_size, num_layers_Encoder, num_layers_Decoder, 
                 num_heads, forward_expansion, dropout, device, max_len):
        
        super(Transformer, self).__init__()
        self.d_ffn = forward_expansion * embedding_size
        self.device = device

        self.Encoder_Embedding_layer = nn.Embedding(src_vocab_size, embedding_size).to(device)
        self.Decoder_Embedding_layer = nn.Embedding(trg_vocab_size, embedding_size).to(device)
        # 旋转位置编码 
        self.Positional_Encoding_layer = Positional_Encoding(embedding_size, max_len, device).to(device)
        # dropout
        self.Dropout_layer =  nn.Dropout(dropout).to(device)  # Dropout
        self.fc = nn.Linear(embedding_size, trg_vocab_size).to(device)  # 最终的全连接层
        self.Encoder_layer = nn.ModuleList([Encoder(embedding_size, num_heads, self.d_ffn, dropout, device) for _ in range(num_layers_Encoder)])
        self.Decoder_layer = nn.ModuleList([Decoder(embedding_size, num_heads, self.d_ffn, dropout, device) for _ in range(num_layers_Decoder)])

    # 掩码
    def Generate_src_mask(self, src):
        # 输入src形状为(batch, seq_len)
        # 输出src_mask形状为(batch, 1， 1， seq_len)，这个mask是在输入时embedding的时候
        # 源掩码：屏蔽填充符（假设填充符索引为0），填充直接返回布尔型的矩阵
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def Generate_trg_mask(self, trg):
        # 输入trg形状为(batch, seq_len)
        # 输出trg_mask形状为(batch, 1， seq_len， seq_len),这个mask是在QK.T之后使用的
        # 目标掩码：屏蔽填充符和未来信息（右边三角填充为0）的布尔型矩阵
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
            
    def forward(self, src, trg):
        src = src.to(self.device)
        trg = trg.to(self.device)
        # 生成掩码
        src_mask = self.Generate_src_mask(src)
        trg_mask = self.Generate_trg_mask(trg)

        # 编码器部分
        src_embed = self.Encoder_Embedding_layer(src)
        src_posi  = self.Positional_Encoding_layer(src_embed)
        src_dropout = self.Dropout_layer(src_posi)
        enc_output = src_dropout
        for enc_layer in self.Encoder_layer:
            enc_output = enc_layer(enc_output, src_mask)

        # 解码器部分
        trg_embed = self.Decoder_Embedding_layer(trg)
        trg_posi  = self.Positional_Encoding_layer(trg_embed)
        trg_dropout = self.Dropout_layer(trg_posi)
        de_output = trg_dropout
        for dec_layer in self.Decoder_layer:
            de_output = dec_layer(src_dropout, trg_dropout, trg_mask, src_mask)

        output = self.fc(de_output)
        return output