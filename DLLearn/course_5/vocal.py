import pickle
import itertools
from collections import Counter
from dataset import WordDataset

zh_vocab_file_path = "/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/chinese.zh/zh_vocab.pkl"
en_vocab_file_path = "/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/english.en/en_vocab.pkl"
src_file = '/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/chinese.zh/chinese.zh'
trg_file = '/root/workspace/MyMLCode/DLLearn/dataset/BOOK_CH_EN/english.en/english.en'

dataset = WordDataset(src_file, trg_file)
zh_tokens = list(itertools.chain.from_iterable(src_tokens for src_tokens, trg_tokens in dataset))
en_tokens = list(itertools.chain.from_iterable(trg_tokens for src_tokens, trg_tokens in dataset))
zh_vocab_counter = Counter(zh_tokens)
en_vocab_counter = Counter(en_tokens)

print(f"zh_vocab_counter_len = {len(zh_vocab_counter)}")
print(f"en_vocab_counter_len = {len(en_vocab_counter)}")

# Minimum frequency for a token to be included in the vocabulary
MIN_FREQ = 1
'''
<pad> ：填充 token ，用于补齐不同长度的序列，使批量数据长度一致 。
<sos> ：序列开始标记（Start Of Sequence ）。
<eos> ：序列结束标记（End Of Sequence ）。
<unk> ：未知 token ，遇到不在词汇表中的词时用它替代 。
这里三个词汇表初始结构一样，后续会分别往 zh_vocab （中文 ）、en_vocab （英文 ）里添加各自语言的 token 
'''
all_vocab = {
    '<pad>':0,
    '<sos>':1,
    '<eos>':2,
    '<unk>':3
}
zh_vocab = {
    '<pad>':0,
    '<sos>':1,
    '<eos>':2,
    '<unk>':3
}
en_vocab = {
    '<pad>':0,
    '<sos>':1,
    '<eos>':2,
    '<unk>':3
}

zh_tot = 4
for index, (token, freq) in enumerate(zh_vocab_counter.items()):
    if freq >= MIN_FREQ:
        zh_vocab.update({token: zh_tot})
        zh_tot += 1    

en_tot = 4
for index, (token, freq) in enumerate(en_vocab_counter.items()):
    if freq >= MIN_FREQ:
        en_vocab.update({token: en_tot})
        en_tot += 1  

print("zh_vocab", list(zh_vocab.items())[:5])
print(f"zh_vocab_len = {len(zh_vocab)}")
print(f"en_vocab_len = {len(en_vocab)}")

with open(zh_vocab_file_path, "wb") as zh_vocab_file:
    pickle.dump(zh_vocab, zh_vocab_file)

with open(en_vocab_file_path, "wb") as en_vocab_file:
    pickle.dump(en_vocab, en_vocab_file)