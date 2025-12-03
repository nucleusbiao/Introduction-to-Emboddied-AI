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


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax和Dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到value
        context = torch.matmul(attn_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.w_o(context)
        return output


class PositionWiseFFN(nn.Module):
    """位置式前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class DecoderLayer(nn.Module):
    """解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 掩码多头自注意力
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 编码器-解码器注意力
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # 编码器和解码器堆栈
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_src_mask(self, src):
        """生成源序列mask（用于padding）"""
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def generate_tgt_mask(self, tgt):
        """生成目标序列mask（用于防止看到未来信息）"""
        batch_size, seq_len = tgt.size()
        # 创建下三角矩阵（包括对角线）
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        # 结合padding mask
        padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        return subsequent_mask.unsqueeze(0) & padding_mask

    def forward(self, src, tgt):
        # 生成mask
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        # 编码器
        src_embedded = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        # 解码器
        tgt_embedded = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 输出层
        output = self.output_linear(decoder_output)
        return output


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

        # 活动描述
        ("Kinder spielen im Park.", "Children are playing in the park."),
        ("Er liest eine Zeitung.", "He is reading a newspaper."),
        ("Sie arbeitet im Büro.", "She works in the office."),
        ("Wir lernen Deutsch.", "We are learning German."),
        ("Sie sehen fern.", "They are watching TV."),

        # 情感表达
        ("Ich liebe Musik.", "I love music."),
        ("Das macht mich glücklich.", "That makes me happy."),
        ("Ich bin müde.", "I am tired."),
        ("Es tut mir leid.", "I am sorry."),
        ("Herzlichen Glückwunsch!", "Congratulations!"),

        # 时间和地点
        ("Es ist drei Uhr.", "It is three o'clock."),
        ("Heute ist Montag.", "Today is Monday."),
        ("Der Bahnhof ist nah.", "The train station is near."),
        ("Wo ist die Toilette?", "Where is the bathroom?"),
        ("Links und dann rechts.", "Left and then right."),

        # 购物和餐饮
        ("Wie viel kostet das?", "How much does this cost?"),
        ("Ich möchte bezahlen.", "I would like to pay."),
        ("Eine Tasse Tee, bitte.", "A cup of tea, please."),
        ("Das Essen schmeckt gut.", "The food tastes good."),
        ("Die Rechnung, bitte.", "The bill, please."),

        # 工作和学习
        ("Ich bin Student.", "I am a student."),
        ("Sie ist Ärztin.", "She is a doctor."),
        ("Wir haben eine Besprechung.", "We have a meeting."),
        ("Der Computer funktioniert nicht.", "The computer is not working."),
        ("Ich schreibe einen Bericht.", "I am writing a report."),

        # 家庭和朋友
        ("Das ist meine Familie.", "This is my family."),
        ("Mein Bruder ist groß.", "My brother is tall."),
        ("Wir treffen uns mit Freunden.", "We are meeting with friends."),
        ("Sie hat braune Haare.", "She has brown hair."),
        ("Er trägt eine Brille.", "He wears glasses."),

        # 旅行和交通
        ("Wann fährt der Zug ab?", "When does the train leave?"),
        ("Ein Flugticket nach Paris.", "A flight ticket to Paris."),
        ("Das Hotel ist voll.", "The hotel is full."),
        ("Ich habe meinen Pass verloren.", "I lost my passport."),
        ("Wo kann ich ein Auto mieten?", "Where can I rent a car?"),

        # 健康和医疗
        ("Ich fühle mich krank.", "I feel sick."),
        ("Kopfschmerzen.", "Headache."),
        ("Wo ist das Krankenhaus?", "Where is the hospital?"),
        ("Ich brauche einen Arzt.", "I need a doctor."),
        ("Nehmen Sie Medikamente?", "Are you taking medication?")
    ]

    # 扩展数据集
    expanded_data = []
    for i in range(5):  # 重复5次以增加数据量
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


def train_transformer():
    """训练Transformer模型"""

    # 准备数据
    train_data, val_data, test_data, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab = prepare_data()

    # 创建数据集
    train_src_texts = [pair[0] for pair in train_data]
    train_tgt_texts = [pair[1] for pair in train_data]
    val_src_texts = [pair[0] for pair in val_data]
    val_tgt_texts = [pair[1] for pair in val_data]

    train_dataset = TranslationDataset(
        train_src_texts, train_tgt_texts, src_tokenizer, tgt_tokenizer,
        src_vocab, tgt_vocab, max_length=30
    )

    val_dataset = TranslationDataset(
        val_src_texts, val_tgt_texts, src_tokenizer, tgt_tokenizer,
        src_vocab, tgt_vocab, max_length=30
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型参数
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(f"\n模型配置:")
    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")

    # 初始化模型（使用较小的参数以便快速训练）
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_length=30,
        dropout=0.1
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练参数
    num_epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model.to(device)

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            # 准备解码器输入和输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            output = model(src, tgt_input)
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = model(src, tgt_input)
                output = output.contiguous().view(-1, output.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)

                loss = criterion(output, tgt_output)
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
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'src_tokenizer': src_tokenizer,
                'tgt_tokenizer': tgt_tokenizer
            }, 'best_transformer_model.pth')
            print(f'  保存最佳模型，验证损失: {best_val_loss:.4f}')

        scheduler.step()
        print('-' * 50)

    return model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab


def translate_sentence(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, sentence, device, max_length=30):
    """翻译单个句子"""
    model.eval()

    # 编码源句子
    src_tensor = src_tokenizer.encode(sentence, src_vocab, max_length).unsqueeze(0).to(device)

    # 初始化目标序列
    tgt_tokens = [tgt_tokenizer.BOS_IDX]

    with torch.no_grad():
        for i in range(max_length):
            tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(device)

            output = model(src_tensor, tgt_tensor)
            next_token = output.argmax(dim=-1)[:, -1].item()

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


def demo_translation(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
    """演示翻译功能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_sentences = [
        "Hallo, wie geht es dir?",
        "Ich heiße Anna",
        "Das Wetter ist schön",
        "Ich trinke Kaffee",
        "Wo ist die Toilette?"
    ]

    print("\n翻译演示:")
    print("=" * 50)

    for sentence in test_sentences:
        translation = translate_sentence(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, sentence, device)
        print(f"德语: {sentence}")
        print(f"英语: {translation}")
        print("-" * 30)


if __name__ == "__main__":
    print("开始训练Transformer翻译模型...")
    print("=" * 60)

    # 训练模型
    model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab = train_transformer()

    # 演示翻译
    demo_translation(model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab)

    print("\n训练完成！")
    print("模型已保存为 'best_transformer_model.pth'")