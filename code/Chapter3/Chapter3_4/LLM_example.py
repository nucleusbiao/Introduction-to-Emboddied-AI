import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from typing import Optional, Tuple


# ==================== 1. Transformer模型定义 ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 确保掩码是bool类型
            if mask.dtype != torch.bool:
                mask = mask.bool()
            attn_scores = attn_scores.masked_fill(mask == False, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 处理掩码形状
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, -1, -1)

        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.w_o(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """基础Transformer模型，支持预训练和微调"""

    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)


# ==================== 2. 掩码语言模型预训练 ====================

class MaskedLanguageModel(nn.Module):
    """掩码语言模型，用于预训练阶段"""

    def __init__(self, transformer: TransformerModel, vocab_size: int):
        super().__init__()
        self.transformer = transformer
        self.vocab_size = vocab_size
        self.mlm_head = nn.Linear(transformer.d_model, vocab_size)

    def create_attention_mask(self, attention_mask):
        """创建注意力掩码"""
        if attention_mask is None:
            return None

        # 简化掩码处理：只使用填充掩码，不使用因果掩码
        batch_size, seq_len = attention_mask.shape
        # 创建双向注意力掩码
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        mask = mask.expand(batch_size, 1, seq_len, seq_len)  # (batch_size, 1, seq_len, seq_len)

        # 转换为bool类型
        mask = mask.bool()
        return mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 创建注意力掩码
        mask = self.create_attention_mask(attention_mask)

        # 创建掩码 - 随机遮盖15%的token
        if labels is not None:
            mask_prob = torch.rand(input_ids.shape, device=input_ids.device) < 0.15
            # 保留特殊token不被遮盖
            special_tokens_mask = (input_ids < 5)  # 前5个token是特殊token
            mask_prob = mask_prob & ~special_tokens_mask

            masked_input = input_ids.clone()
            # 80%的时间：用[MASK]替换
            mask_token = 4  # [MASK] token ID
            mask_mask = mask_prob & (torch.rand(mask_prob.shape, device=input_ids.device) < 0.8)
            masked_input[mask_mask] = mask_token

            # 10%的时间：用随机token替换
            random_mask = mask_prob & (torch.rand(mask_prob.shape, device=input_ids.device) < 0.5)
            random_tokens = torch.randint(5, self.vocab_size, input_ids.shape, device=input_ids.device)
            masked_input[random_mask] = random_tokens[random_mask]
            # 剩余10%的时间保持原样

            transformer_output = self.transformer(masked_input, mask)
        else:
            transformer_output = self.transformer(input_ids, mask)

        logits = self.mlm_head(transformer_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
            masked_lm_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return logits, masked_lm_loss

        return logits


# ==================== 3. LoRA高效微调 ====================

class LoRALayer(nn.Module):
    """LoRA适配器层"""

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA参数 - 只训练A和B
        self.lora_A = nn.Parameter(torch.randn(base_layer.in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, base_layer.out_features))

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output


def apply_lora_to_transformer(model: nn.Module, rank: int = 4):
    """将LoRA应用到Transformer的线性层"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'head' not in name:
            setattr(model, name, LoRALayer(module, rank))
        else:
            apply_lora_to_transformer(module, rank)


# ==================== 4. 数据集和训练流程 ====================

class TextDataset(Dataset):
    """简单的文本数据集"""

    def __init__(self, texts, vocab, max_length=512):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.lower().split()[:self.max_length]
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

        padded = token_ids + [self.vocab['[PAD]']] * (self.max_length - len(token_ids))
        attention_mask = [1] * len(token_ids) + [0] * (self.max_length - len(token_ids))

        return {
            'input_ids': torch.tensor(padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded, dtype=torch.long)
        }


class Pretrainer:
    """预训练管理器"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def pretrain_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            _, loss = self.model(input_ids, attention_mask, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


class FineTuner:
    """微调管理器"""

    def __init__(self, base_model, num_classes, device='cuda', use_lora=False):
        self.device = device
        self.base_model = base_model

        if use_lora:
            apply_lora_to_transformer(base_model, rank=4)
            trainable_params = []
            for name, param in base_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            print(f"使用LoRA微调，可训练参数数量: {len(trainable_params)}")
        else:
            for param in base_model.parameters():
                param.requires_grad = True

        self.classifier = nn.Linear(base_model.d_model, num_classes).to(device)

        # 差分学习率
        base_params = []
        head_params = []
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                base_params.append(param)
        for param in self.classifier.parameters():
            head_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': base_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-4}
        ], weight_decay=0.01)

    def create_attention_mask(self, attention_mask):
        """创建注意力掩码"""
        if attention_mask is None:
            return None
        batch_size, seq_len = attention_mask.shape
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        mask = mask.expand(batch_size, 1, seq_len, seq_len)  # (batch_size, 1, seq_len, seq_len)
        mask = mask.bool()  # 转换为bool类型
        return mask

    def finetune_epoch(self, dataloader):
        self.base_model.train()
        self.classifier.train()
        total_loss, total_acc = 0, 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            mask = self.create_attention_mask(attention_mask)

            self.optimizer.zero_grad()
            outputs = self.base_model(input_ids, mask)

            # 使用平均池化
            pooled_output = outputs.mean(dim=1)
            logits = self.classifier(pooled_output)

            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_acc += (preds == labels).float().mean().item()

        return total_loss / len(dataloader), total_acc / len(dataloader)


# ==================== 5. 演示代码 ====================

def demonstrate_pretraining_finetuning():
    """演示完整的预训练和微调流程"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建词汇表
    vocab = {
        '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
        **{f'word_{i}': i + 5 for i in range(50)}  # 更小的词汇表
    }
    vocab_size = len(vocab)

    # 1. 初始化基础Transformer模型
    print("初始化基础Transformer模型...")
    base_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,  # 更小的前馈网络
        max_seq_len=32
    )

    # 2. 预训练阶段（掩码语言模型）
    print("\n=== 预训练阶段 ===")
    mlm_model = MaskedLanguageModel(base_model, vocab_size)
    pretrainer = Pretrainer(mlm_model, device)

    # 模拟预训练数据
    pretrain_texts = [
                         "机器人 带着 球 冲向 球门",
                         "球员 在 足球场 上 奔跑",
                         "这 是 一个 关于 机器人 的 演示",
                         "预训练 模型 学习 通用 知识"
                     ] * 5  # 更少的数据

    pretrain_dataset = TextDataset(pretrain_texts, vocab, max_length=16)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=2, shuffle=True)

    # 模拟预训练
    print("开始预训练...")
    for epoch in range(2):
        loss = pretrainer.pretrain_epoch(pretrain_loader)
        print(f"预训练 Epoch {epoch + 1}, Loss: {loss:.4f}")

    # 3. 微调阶段（分类任务）
    print("\n=== 微调阶段 ===")

    # 模拟下游任务数据
    finetune_texts = [
                         "机器人 足球 比赛",
                         "技术 演示 文本",
                         "运动 相关 描述",
                         "其他 类别 文本"
                     ] * 3

    finetune_labels = [0, 1, 2, 3] * 3

    class FineTuneDataset(Dataset):
        def __init__(self, texts, labels, vocab, max_length=16):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            tokens = text.lower().split()[:self.max_length]
            token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

            padded = token_ids + [self.vocab['[PAD]']] * (self.max_length - len(token_ids))
            attention_mask = [1] * len(token_ids) + [0] * (self.max_length - len(token_ids))

            return {
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    finetune_dataset = FineTuneDataset(finetune_texts, finetune_labels, vocab)
    finetune_loader = DataLoader(finetune_dataset, batch_size=2, shuffle=True)

    # 全参数微调
    print("--- 全参数微调 ---")
    full_finetuner = FineTuner(base_model, num_classes=4, device=device, use_lora=False)

    for epoch in range(2):
        loss, acc = full_finetuner.finetune_epoch(finetune_loader)
        print(f"全参数微调 Epoch {epoch + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")

    # LoRA高效微调
    print("\n--- LoRA高效微调 ---")
    base_model_lora = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_len=32
    )

    lora_finetuner = FineTuner(base_model_lora, num_classes=4, device=device, use_lora=True)

    for epoch in range(2):
        loss, acc = lora_finetuner.finetune_epoch(finetune_loader)
        print(f"LoRA微调 Epoch {epoch + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")


# ==================== 6. Scaling Laws 演示 ====================

def demonstrate_scaling_laws():
    """演示模型规模对性能的影响"""
    print("\n=== Scaling Laws 演示 ===")

    model_sizes = [1e6, 10e6, 100e6, 1e9]
    performances = [0.3, 0.5, 0.7, 0.85]

    print("模型规模 vs 性能:")
    for size, perf in zip(model_sizes, performances):
        if size >= 1e9:
            size_str = f"{size / 1e9:.1f}B"
        else:
            size_str = f"{size / 1e6:.0f}M"
        print(f"  {size_str}参数: {perf:.2f} 性能")

    print("\n涌现能力演示:")
    critical_sizes = [100e6, 500e6, 1e9, 10e9]
    emergent_abilities = [
        "基础语言理解",
        "简单推理",
        "上下文学习",
        "复杂推理和代码生成"
    ]

    for size, ability in zip(critical_sizes, emergent_abilities):
        size_str = f"{size / 1e9:.0f}B" if size >= 1e9 else f"{size / 1e6:.0f}M"
        print(f"  规模达到 {size_str}: 出现'{ability}'能力")


if __name__ == "__main__":
    try:
        # 运行完整演示
        demonstrate_pretraining_finetuning()
        demonstrate_scaling_laws()

        print("\n" + "=" * 50)
        print("预训练与大模型演示完成!")
        print("核心概念总结:")
        print("1. 预训练: 通过掩码语言模型在海量数据上学习通用知识")
        print("2. 微调: 在小规模任务数据上适配特定任务")
        print("3. LoRA: 高效微调技术，大幅减少可训练参数")
        print("4. Scaling Laws: 模型规模与性能的正相关关系")
        print("5. 涌现: 模型规模突破临界点后出现的新能力")
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        print("这可能是由于硬件限制，但核心概念已经展示")