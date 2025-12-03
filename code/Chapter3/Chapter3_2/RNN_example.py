import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import os
import urllib.request
import tarfile
import warnings

warnings.filterwarnings('ignore')


class Encoder(nn.Module):
    """RNN编码器"""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 双向GRU输出合并

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_size]

        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, seq_len, hidden_size * 2] (双向)
        # hidden: [num_layers * 2, batch_size, hidden_size]

        # 合并双向隐藏状态
        hidden = self._merge_bidirectional_hidden(hidden)

        return outputs, hidden

    def _merge_bidirectional_hidden(self, hidden):
        """合并双向GRU的隐藏状态"""
        # hidden: [num_layers * 2, batch_size, hidden_size]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden_forward = hidden[:, 0, :, :]  # [num_layers, batch_size, hidden_size]
        hidden_backward = hidden[:, 1, :, :]  # [num_layers, batch_size, hidden_size]

        # 合并最后层的隐藏状态用于解码器初始化
        merged_hidden = torch.cat([hidden_forward[-1], hidden_backward[-1]], dim=1)  # [batch_size, hidden_size * 2]
        merged_hidden = self.fc(merged_hidden).unsqueeze(0)  # [1, batch_size, hidden_size]

        return merged_hidden


class Attention(nn.Module):
    """注意力机制"""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.energy = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        # hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size * 2]

        batch_size, seq_len, hidden_dim = encoder_outputs.shape

        # 重复隐藏状态以匹配序列长度
        hidden_repeated = hidden.transpose(0, 1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]

        # 计算能量分数
        energy_input = torch.cat([hidden_repeated, encoder_outputs], dim=2)  # [batch_size, seq_len, hidden_size * 3]
        energy = torch.tanh(self.energy(energy_input))  # [batch_size, seq_len, hidden_size]
        attention_scores = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        # 应用源序列mask
        if src_mask is not None:
            attention_scores = attention_scores.masked_fill(src_mask == 0, -1e9)

        # Softmax得到注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len]

        # 计算上下文向量
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size * 2]

        return context_vector, attention_weights


class Decoder(nn.Module):
    """RNN解码器"""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(embed_size + hidden_size * 2, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 3 + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs, src_mask=None):
        # x: [batch_size] (单个token)
        # hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size * 2]

        x = x.unsqueeze(1)  # [batch_size, 1]

        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embed_size]

        # 计算注意力
        context_vector, attention_weights = self.attention(hidden, encoder_outputs, src_mask)

        # 组合嵌入和上下文向量作为RNN输入
        rnn_input = torch.cat([embedded, context_vector], dim=2)  # [batch_size, 1, embed_size + hidden_size * 2]

        # RNN前向传播
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch_size, 1, hidden_size]
        # hidden: [1, batch_size, hidden_size]

        # 准备全连接层输入
        output = output.squeeze(1)  # [batch_size, hidden_size]
        embedded = embedded.squeeze(1)  # [batch_size, embed_size]
        context_vector = context_vector.squeeze(1)  # [batch_size, hidden_size * 2]

        # 预测下一个token
        prediction = self.fc_out(torch.cat([output, embedded, context_vector], dim=1))  # [batch_size, vocab_size]

        return prediction, hidden, attention_weights


class Seq2Seq(nn.Module):
    """基于RNN的序列到序列模型"""

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        # 存储输出
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)

        # 第一个输入是<BOS> token
        input = tgt[:, 0]

        # 生成源序列mask
        src_mask = self.generate_src_mask(src)

        for t in range(1, tgt_len):
            # 解码器前向传播
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, src_mask)

            # 存储输出
            outputs[:, t] = output

            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # 获取下一个输入（真实目标或预测）
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

    def generate_src_mask(self, src):
        """生成源序列mask（用于padding）"""
        return (src != 0)  # [batch_size, src_len]


def train_rnn_model():
    """训练RNN模型"""

    # 准备数据
    train_data, val_data, test_data, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab = prepare_data()

    # 创建数据集
    train_src_texts = [pair[0] for pair in train_data]
    train_tgt_texts = [pair[1] for pair in train_data]
    val_src_texts = [pair[0] for pair in val_data]
    val_tgt_texts = [pair[1] for pair in val_data]

    train_dataset = TranslationDataset(
        train_src_texts, train_tgt_texts, src_tokenizer, tgt_tokenizer,
        src_vocab, tgt_vocab, max_length=20  # 减少序列长度
    )

    val_dataset = TranslationDataset(
        val_src_texts, val_tgt_texts, src_tokenizer, tgt_tokenizer,
        src_vocab, tgt_vocab, max_length=20
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 减少batch_size
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 模型参数
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    embed_size = 64  # 减少嵌入维度
    hidden_size = 128  # 减少隐藏层维度
    num_layers = 1  # 减少层数
    dropout = 0.1

    print(f"\n模型配置:")
    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")
    print(f"嵌入维度: {embed_size}")
    print(f"隐藏层维度: {hidden_size}")
    print(f"层数: {num_layers}")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 打印模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数数量: {count_parameters(model):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练参数
    num_epochs = 20  # 减少训练轮数
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            output = model(src, tgt, teacher_forcing_ratio=0.5)
            # output: [batch_size, tgt_len, tgt_vocab_size]

            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)  # 跳过第一个token (<BOS>)
            tgt = tgt[:, 1:].contiguous().view(-1)  # 跳过第一个token

            loss = criterion(output, tgt)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)

                output = model(src, tgt, teacher_forcing_ratio=0)  # 推理时不使用teacher forcing
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                tgt = tgt[:, 1:].contiguous().view(-1)

                loss = criterion(output, tgt)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'src_tokenizer': src_tokenizer,
                'tgt_tokenizer': tgt_tokenizer,
                'model_config': {
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'embed_size': embed_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                }
            }, 'best_rnn_model.pth')
            print(f'  保存最佳模型，验证损失: {best_val_loss:.4f}')

        scheduler.step()
        print('-' * 50)

    return model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab


def translate_sentence_rnn(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, sentence, device, max_length=20):
    """使用RNN模型翻译单个句子"""
    model.eval()

    # 编码源句子
    src_tensor = src_tokenizer.encode(sentence, src_vocab, max_length).unsqueeze(0).to(device)

    with torch.no_grad():
        # 编码器前向传播
        encoder_outputs, hidden = model.encoder(src_tensor)

        # 生成源序列mask
        src_mask = model.generate_src_mask(src_tensor)

        # 初始化目标序列
        tgt_tokens = [tgt_tokenizer.BOS_IDX]

        for i in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens[-1]], dtype=torch.long).to(device)

            output, hidden, attention_weights = model.decoder(tgt_tensor, hidden, encoder_outputs, src_mask)

            next_token = output.argmax(1).item()

            if next_token == tgt_tokenizer.EOS_IDX:
                break

            tgt_tokens.append(next_token)

    # 将token ID转换回文本
    reverse_vocab = {v: k for k, v in tgt_vocab.items()}
    translated_words = []
    for token_id in tgt_tokens[1:]:  # 跳过BOS
        if token_id in reverse_vocab:
            translated_words.append(reverse_vocab[token_id])

    return ' '.join(translated_words)


def demo_translation_rnn(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
    """演示RNN翻译功能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_sentences = [
        "Hallo, wie geht es dir?",
        "Ich heiße Anna",
        "Das Wetter ist schön",
        "Ich trinke Kaffee",
        "Wo ist die Toilette?"
    ]

    print("\nRNN模型翻译演示:")
    print("=" * 50)

    for sentence in test_sentences:
        translation = translate_sentence_rnn(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, sentence,
                                             device)
        print(f"德语: {sentence}")
        print(f"英语: {translation}")
        print("-" * 30)


# 保留原有的 SimpleTokenizer, TranslationDataset, create_realistic_translation_data, prepare_data 函数
# 这些函数不需要修改

class SimpleTokenizer:
    """简单的分词器"""

    def __init__(self):
        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3

    def tokenize(self, text):
        """简单的按空格分词"""
        return text.lower().split()

    def build_vocab(self, texts, min_freq=1):
        """构建词汇表"""
        word_freq = {}
        for text in texts:
            for word in self.tokenize(text):
                word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 构建词汇表
        vocab = {'<pad>': self.PAD_IDX, '<bos>': self.BOS_IDX,
                 '<eos>': self.EOS_IDX, '<unk>': self.UNK_IDX}

        for word, freq in sorted_words:
            if freq >= min_freq and word not in vocab:
                vocab[word] = len(vocab)

        return vocab

    def encode(self, text, vocab, max_length=None):
        """将文本编码为token ID"""
        tokens = [self.BOS_IDX]
        for word in self.tokenize(text):
            tokens.append(vocab.get(word, self.UNK_IDX))
        tokens.append(self.EOS_IDX)

        if max_length:
            if len(tokens) < max_length:
                tokens.extend([self.PAD_IDX] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
                tokens[-1] = self.EOS_IDX

        return torch.tensor(tokens, dtype=torch.long)


class TranslationDataset(Dataset):
    """翻译数据集"""

    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer,
                 src_vocab, tgt_vocab, max_length=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_encoded = self.src_tokenizer.encode(
            self.src_texts[idx], self.src_vocab, self.max_length)
        tgt_encoded = self.tgt_tokenizer.encode(
            self.tgt_texts[idx], self.tgt_vocab, self.max_length)

        return src_encoded, tgt_encoded


def create_realistic_translation_data():
    """创建真实的德语-英语翻译数据"""
    german_english_pairs = [
        # 日常对话
        ("Hallo, wie geht es dir?", "Hello, how are you?"),
        ("Ich heiße Anna.", "My name is Anna."),
        ("Woher kommst du?", "Where are you from?"),
        ("Ich komme aus Berlin.", "I am from Berlin."),
        ("Wie alt bist du?", "How old are you?"),
        ("Ich bin zwanzig Jahre alt.", "I am twenty years old."),

        # 日常生活
        ("Das Wetter ist schön heute.", "The weather is nice today."),
        ("Ich gehe einkaufen.", "I am going shopping."),
        ("Was möchtest du essen?", "What would you like to eat?"),
        ("Ich trinke Kaffee.", "I drink coffee."),
        ("Das Buch ist interessant.", "The book is interesting."),

        # 更简单的句子用于初始测试
        ("Hallo", "Hello"),
        ("Danke", "Thank you"),
        ("Ja", "Yes"),
        ("Nein", "No"),
    ]

    # 扩展数据集
    expanded_data = []
    for i in range(3):  # 减少重复次数
        for pair in german_english_pairs:
            expanded_data.append(pair)

    return expanded_data


def prepare_data():
    """准备训练数据"""
    print("准备翻译数据...")

    # 创建真实翻译数据
    all_data = create_realistic_translation_data()

    # 分割数据集
    random.shuffle(all_data)
    train_size = int(0.8 * len(all_data))
    val_size = int(0.1 * len(all_data))

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 分离源语言和目标语言文本
    src_texts = [pair[0] for pair in train_data]
    tgt_texts = [pair[1] for pair in train_data]

    # 初始化分词器
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    # 构建词汇表
    print("构建词汇表...")
    src_vocab = src_tokenizer.build_vocab(src_texts, min_freq=1)
    tgt_vocab = tgt_tokenizer.build_vocab(tgt_texts, min_freq=1)

    print(f"德语词汇表大小: {len(src_vocab)}")
    print(f"英语词汇表大小: {len(tgt_vocab)}")

    return (train_data, val_data, test_data,
            src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab)


if __name__ == "__main__":
    print("开始训练RNN翻译模型...")
    print("=" * 60)

    # 训练模型
    model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab = train_rnn_model()

    # 演示翻译
    demo_translation_rnn(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab)

    print("\n训练完成！")
    print("模型已保存为 'best_rnn_model.pth'")