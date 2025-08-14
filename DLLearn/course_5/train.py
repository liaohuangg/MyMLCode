import torch
import pickle
import torch.nn as nn
from utils import translate_sentence
from torch.utils.data import DataLoader
from dataset import NumberDataset
from model import Transformer
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# hyperparameters 1
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-4
max_len = 120

# get vocabulary
zh_vocab_file_path = "/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/chinese.zh/zh_vocab.pkl"
en_vocab_file_path = "/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/english.en/en_vocab.pkl"

# Load the vocabulary files
with open(zh_vocab_file_path, "rb") as zh_vocab_file:
    zh_vocab = pickle.load(zh_vocab_file)

with open(en_vocab_file_path, "rb") as en_vocab_file:
    en_vocab = pickle.load(en_vocab_file)

# print("zh_vocab", list(zh_vocab.items())[:10])
# print("en_vocab", list(en_vocab.items())[:10])
'''
zh_vocab [('<pad>', 0), ('<sos>', 1), ('<eos>', 2), ('<unk>', 3), ('1929年', 4), ('还是', 5), ('1989年', 6), ('?', 7), ('\n', 8), ('巴黎', 9)]
en_vocab [('<pad>', 0), ('<sos>', 1), ('<eos>', 2), ('<unk>', 3), ('1929', 4), ('or', 5), ('1989', 6), ('?', 7), ('\n', 8), ('PARIS', 9)]
'''
# get the index: token mapping 用来转义句子
zh_ivocab = {index: token for token, index in zh_vocab.items()}
en_ivocab = {index: token for token, index in en_vocab.items()}
# print("zh_ivocab", list(zh_ivocab.items())[:10])
# print("en_ivocab", list(en_ivocab.items())[:10])
'''
zh_ivocab [(0, '<pad>'), (1, '<sos>'), (2, '<eos>'), (3, '<unk>'), (4, '1929年'), (5, '还是'), (6, '1989年'), (7, '?'), (8, '\n'), (9, '巴黎')]
en_ivocab [(0, '<pad>'), (1, '<sos>'), (2, '<eos>'), (3, '<unk>'), (4, '1929'), (5, 'or'), (6, '1989'), (7, '?'), (8, '\n'), (9, 'PARIS')]
'''

# get the dataset
src_file = '/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/chinese.zh/chinese.zh'
trg_file = '/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/english.en/english.en'

# Load the dataset
dataset = NumberDataset(src_file, trg_file, zh_vocab, en_vocab, max_len=max_len)
train_loader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"--------data loaded--------")

# parameters
src_vocab_size = len(zh_vocab)
trg_vocab_size = len(en_vocab)
# 输入维度（通常是 embedding_size ）
embedding_size = 512
num_heads = 8
num_layers_FFN = 3
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 4
src_pad_idx = 0
trg_pad_idx = 0

# get model
model = Transformer(
    src_vocab_size, 
    trg_vocab_size,
    embedding_size, 
    num_encoder_layers, 
    num_decoder_layers, 
    num_heads, 
    forward_expansion, 
    dropout, 
    DEVICE, 
    max_len
).to(DEVICE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充部分的损失
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
model.train()
for epoch in range(EPOCHS):

    print(f"[Epoch {epoch} / {EPOCHS}]")
    
    # 推理模式
    model.eval()
    sentence = "我也不知道你在说什么。"
    # 使用模型进行翻译
    translated_sentence = translate_sentence(
        model, sentence,  zh_vocab, en_ivocab, DEVICE, max_len=50
    )
    print("翻译的句子如下：", translated_sentence)


    # 训练模式
    model.train()
    losses = []

    for i, (src, trg) in enumerate(train_loader):
        input = src.to(DEVICE)
        target = trg.to(DEVICE)

        # 为了实现 teacher forcing，目标序列的输入是去掉最后一个词的
        # 计算损失时，目标序列从第二个词开始（即预测下一个词）
        # output形状: (batch_size, seq_length-1, tgt_vocab_size)
        # 目标形状: (batch_size, seq_length-1)
        output = model(input, target[:, :-1])  # 去掉最后一个词
        # (batch_size, seq_len-1, embedding_size) -> (batch_size*(seq_len-1), embedding_size)
        output = output.reshape(-1, output.shape[2])
        # 去掉第一个词， (batch_size, seq_len-1) -> (batch_size*(seq_len-1))
        target = target[:, 1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if i % 100 == 0:
            print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch, EPOCHS, i//100, loss.item()))

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
