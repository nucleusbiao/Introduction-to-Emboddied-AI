# 《具身智能导论——以人形机器人足球赛为视角》案例代码库

欢迎使用《具身智能导论——以人形机器人足球赛为视角》的配套案例代码库。本代码库包含教材中涉及的核心算法实现和实验案例，旨在助读者理解具身智能的关键概念和实际应用。

## 代码结构

代码库按照教材章节组织，各章节重点内容如下：

```
Introduction-to-Emboddied-AI/
├── code/
│   ├── Chapter1/           # 第一章：基础知识
│   │   └── Chapter1_1/
│   │       └── 3_classification.py  # 分类算法示例
│   ├── Chapter2/           # 第二章：感知与定位
│   │   ├── Chapter2_1/
│   │   │   └── task2.py            # 足球检测基础实现
│   │   └── Chapter2_3/
│   │       ├── robot_localization.py  # 机器人定位模块
│   │       └── camPosePF.py           # 基于粒子滤波的相机位姿估计
│   ├── Chapter3/           # 第三章：注意力机制与大语言模型应用
│   │   ├── Chapter3_1/
│   │   │   └── self_attention.py      # 自注意力机制基础实现
│   │   ├── Chapter3_2/
│   │   │   └── RNN_example.py         # 循环神经网络应用示例
│   │   ├── Chapter3_3/
│   │   │   └── ViT_Example.py         # 视觉Transformer（ViT）应用示例
│   │   └── Chapter3_4/
│   │       └── LLM_example.py         # 大语言模型（LLM）接口与应用示例
│   └── Chapter4/           # 第四章：强化学习基础
│       └── Chapter4_1/
│           └── rlBasics.py           # 强化学习基础算法实现
```

## 主要模块说明

### 1. 分类算法 (Chapter1/Chapter1_1/3_classification.py)

实现了一个基于深度神经网络的三分类任务示例，包括：
- 生成分类数据集并可视化
- 数据集分割与预处理
- 深度神经网络模型构建与训练
- 模型性能评估与结果可视化

```python
# 主要功能示例
def example_training():
    # 创建数据集
    X, Y = create_better_3class_data(n_samples, n_features, n_classes)
    
    # 分割数据集
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, test_size=0.2)
    
    # 创建并训练模型
    nn = DeepNeuralNetwork(layer_dims, learning_rate=0.1, activation='relu')
    costs = nn.train(X_train, Y_train, epochs=1000, verbose=True)
    
    # 评估模型
    train_accuracy = nn.accuracy(X_train, Y_train)
    test_accuracy = nn.accuracy(X_test, Y_test)
```

### 2. 目标检测与定位 (Chapter2/Chapter2_1 ~ Chapter2_3)

#### 2.1 足球检测 (Chapter2/Chapter2_1/task2.py)
实现了基于ONNX模型的足球检测功能，包括：
- 模型加载与初始化
- 图像预处理与推理
- 检测结果后处理
- 检测结果可视化

```python
# 主要功能示例
detector = SimpleFootballDetector(model_path, confidence_threshold=0.25)
detection = detector.inference(image)
result_image = detector.draw_detection(image, detection, world_position)
```

#### 2.2 机器人定位 (Chapter2/Chapter2_3/)
包含机器人定位相关的核心模块：
- `robot_localization.py`：实现相机模型和场地标记检测
  - 相机内参管理与畸变校正
  - 图像坐标与世界坐标转换
  - 多类别场地标记检测
- `camPosePF.py`：实现基于粒子滤波的位姿估计
  - 粒子群初始化与更新
  - 权重计算与重采样
  - 位姿估计与收敛判断

```python
# 定位功能示例
result = locate_robot(markers, constraints, max_iter=50, 
                     max_residual=0.5, conv_threshold=0.1, noise_level=0.01)
if result.success:
    print(f"定位成功: x={result.pose.x}, y={result.pose.y}, theta={result.pose.theta}")
```

### 3. 注意力机制与大语言模型应用 (Chapter3/Chapter3_1 ~ Chapter3_4)
第三章聚焦具身智能中的核心表征学习技术，通过注意力机制、序列模型、视觉Transformer和大语言模型的实现与应用，解决机器人足球场景中的感知理解、序列决策等问题。

#### 3.1 自注意力机制基础 (Chapter3/Chapter3_1/self_attention.py)
实现自注意力机制的核心原理与简化版实现，适配机器人足球场景中的序列信息处理（如动作序列、感知时序数据），包括：
- 缩放点积注意力（Scaled Dot-Product Attention）核心计算
- 多头注意力（Multi-Head Attention）并行特征提取
- 位置编码（Positional Encoding）时序/空间信息注入
- 注意力权重可视化与解释
- 适配机器人动作序列、场地状态序列的处理接口

```python
# 自注意力机制核心实现示例
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # 计算注意力分数：(batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # 掩码处理（可选，用于屏蔽无效位置）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 注意力权重归一化
        attn_weights = torch.softmax(scores, dim=-1)
        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# 多头注意力示例
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 线性投影 + 多头拆分
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 缩放点积注意力计算
        attn_output, attn_weights = ScaledDotProductAttention()(q, k, v, mask)
        # 多头拼接 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.w_o(attn_output)
        return output, attn_weights

# 机器人动作序列处理示例
def example_robot_action_attention():
    # 模拟机器人5个时间步的动作序列（每个动作向量维度为64）
    batch_size, seq_len, d_model = 2, 5, 64
    action_seq = torch.randn(batch_size, seq_len, d_model)
    # 初始化多头注意力模型
    multi_head_attn = MultiHeadAttention(d_model=64, num_heads=8)
    # 位置编码
    pos_encoder = PositionalEncoding(d_model=64, max_len=seq_len)
    action_seq_with_pos = pos_encoder(action_seq)
    # 自注意力计算（q=k=v，即自注意力）
    attn_output, attn_weights = multi_head_attn(action_seq_with_pos, action_seq_with_pos, action_seq_with_pos)
    # 可视化注意力权重
    visualize_attention_weights(attn_weights[0], seq_len=5, title="机器人动作序列注意力权重")
    print(f"输入序列形状: {action_seq.shape}")
    print(f"注意力输出形状: {attn_output.shape}")
```

#### 3.2 循环神经网络应用 (Chapter3/Chapter3_2/RNN_example.py)
实现循环神经网络（RNN）及其变体（LSTM、GRU），用于机器人足球场景中的时序数据建模（如状态预测、动作序列生成），包括：
- 基础RNN、LSTM、GRU模型构建
- 场地状态时序预测（如足球位置、队友/对手运动轨迹预测）
- 动作序列生成（基于历史动作预测下一阶段动作）
- 时序数据预处理与序列长度适配
- 模型训练与预测结果可视化

```python
# RNN/LSTM/GRU模型定义示例
class SequencePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, model_type='rnn'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        # 选择模型类型
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be 'rnn', 'lstm' or 'gru'")
        
        # 输出层（预测下一个时间步的状态）
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len, input_dim)
        rnn_out, hidden = self.rnn(x, hidden)
        # 取最后一个时间步的输出用于预测
        out = self.fc(rnn_out[:, -1, :])
        return out, hidden

# 足球位置时序预测示例
def example_football_position_prediction():
    # 1. 生成模拟数据：足球在场地的(x,y)坐标时序序列
    seq_len = 10  # 历史序列长度
    pred_len = 1  # 预测未来1个时间步
    n_samples = 1000  # 样本数
    X, y = generate_football_trajectory_data(n_samples, seq_len, pred_len)
    
    # 2. 数据预处理：划分训练集/测试集，转换为Tensor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # 3. 初始化模型（输入维度2：x,y坐标；输出维度2：预测的x,y坐标）
    model = SequencePredictor(input_dim=2, hidden_dim=32, output_dim=2, num_layers=2, model_type='lstm')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 模型训练
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, _ = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_output, _ = model(X_test)
                test_loss = criterion(test_output, y_test)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_output.item():.4f}")
    
    # 5. 预测结果可视化
    model.eval()
    with torch.no_grad():
        sample_idx = 0
        sample_seq = X_test[sample_idx:sample_idx+1]
        pred_pos = model(sample_seq)[0].numpy()
        true_pos = y_test[sample_idx].numpy()
        # 可视化历史轨迹、真实位置、预测位置
        visualize_trajectory_prediction(sample_seq[0].numpy(), true_pos, pred_pos)
```

#### 3.3 视觉Transformer（ViT）应用 (Chapter3/Chapter3_3/ViT_Example.py)
实现视觉Transformer（Vision Transformer, ViT）模型，用于机器人足球场景中的图像理解任务（如场地场景分类、目标检测辅助、动作状态识别），包括：
- ViT核心模块实现（Patch Embedding、Transformer Encoder、分类头）
- 场地场景分类（如进攻区、防守区、中场区识别）
- 目标状态识别（如足球是否在控制中、机器人动作状态分类）
- 预训练权重加载与微调适配
- 图像输入预处理与注意力热力图可视化

```python
# ViT核心模块定义示例
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 图像分块与嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch_size, in_chans, img_size, img_size)
        batch_size = x.shape[0]
        x = self.proj(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 num_heads=12, num_layers=12, num_classes=3, dropout=0.1):
        super().__init__()
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # 类别token与位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, 
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x, return_attn=False):
        # x: (batch_size, in_chans, img_size, img_size)
        batch_size = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer Encoder（获取最后一层注意力权重用于可视化）
        if return_attn:
            attn_weights = []
            for layer in self.transformer_encoder.layers:
                x, attn = layer.self_attn(x, x, x, need_weights=True)
                x = layer.norm1(x + layer.dropout1(attn))
                x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.linear1(x)))))
                attn_weights.append(attn)
            x = self.norm(x)
            cls_output = x[:, 0]  # 取类别token的输出
            logits = self.fc(cls_output)
            return logits, attn_weights
        else:
            x = self.transformer_encoder(x)
            x = self.norm(x)
            cls_output = x[:, 0]
            logits = self.fc(cls_output)
            return logits

# 场地场景分类示例
def example_pitch_scene_classification():
    # 1. 配置参数
    img_size = 224
    patch_size = 16
    num_classes = 3  # 场景类别：进攻区、防守区、中场区
    batch_size = 8
    epochs = 20
    
    # 2. 加载数据集（模拟机器人相机拍摄的场地图像数据集）
    train_loader, test_loader = get_pitch_scene_dataset(batch_size=batch_size, img_size=img_size)
    
    # 3. 初始化ViT模型
    model = ViT(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. 模型训练与微调
    train_vit_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    
    # 5. 模型推理与注意力可视化
    model.eval()
    with torch.no_grad():
        # 取一张测试图像
        test_img, test_label = next(iter(test_loader))
        test_img = test_img[:1]  # 单张图像推理
        # 推理并返回注意力权重
        logits, attn_weights = model(test_img, return_attn=True)
        pred_label = torch.argmax(logits, dim=1).item()
        true_label = test_label[0].item()
        
        # 可视化原始图像与注意力热力图（最后一层多头注意力的平均）
        visualize_vit_attention(test_img[0], attn_weights[-1][0].mean(dim=0), 
                               patch_size=patch_size, true_label=true_label, pred_label=pred_label)
        print(f"真实场景类别: {true_label}, 预测场景类别: {pred_label}")
```

#### 3.4 大语言模型（LLM）接口与应用 (Chapter3/Chapter3_4/LLM_example.py)
实现大语言模型（LLM）的调用接口与具身智能场景应用，聚焦机器人足球中的自然语言交互、策略生成、指令理解等任务，包括：
- 主流LLM（如GPT、LLaMA、ChatGLM）的API/本地调用封装
- 机器人指令理解（自然语言→动作指令映射）
- 比赛策略生成（基于场地状态描述生成战术建议）
- 多轮对话交互（与机器人进行比赛相关的自然语言沟通）
- prompt工程优化（适配具身场景的指令设计）

```python
# LLM调用封装类示例
class FootballLLMClient:
    def __init__(self, model_name='chatglm', api_key=None, local_model_path=None):
        """
        初始化LLM客户端
        :param model_name: 模型名称（支持'chatglm', 'gpt', 'llama'）
        :param api_key: API密钥（远程调用时使用）
        :param local_model_path: 本地模型路径（本地部署时使用）
        """
        self.model_name = model_name
        self.api_key = api_key
        self.local_model_path = local_model_path
        self.model = self._load_model()
    
    def _load_model(self):
        """加载模型（远程API或本地模型）"""
        if self.model_name == 'chatglm' and self.local_model_path:
            # 本地加载ChatGLM模型
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.local_model_path, trust_remote_code=True).half().cuda()
            model.eval()
            return (tokenizer, model)
        elif self.model_name == 'gpt' and self.api_key:
            # 远程调用GPT API（示例）
            import openai
            openai.api_key = self.api_key
            return openai.ChatCompletion
        elif self.model_name == 'llama' and self.local_model_path:
            # 本地加载LLaMA模型（需依赖transformers库）
            from transformers import LlamaTokenizer, LlamaForCausalLM
            tokenizer = LlamaTokenizer.from_pretrained(self.local_model_path)
            model = LlamaForCausalLM.from_pretrained(self.local_model_path).half().cuda()
            model.eval()
            return (tokenizer, model)
        else:
            raise ValueError("请提供有效的模型配置（API密钥或本地模型路径）")
    
    def generate_strategy(self, field_status, max_tokens=500):
        """
        基于场地状态生成比赛策略
        :param field_status: 场地状态描述（自然语言）
        :return: 生成的战术策略
        """
        # 构建具身场景Prompt
        prompt = f"""你是一个人形机器人足球赛的战术顾问，基于以下场地状态，生成简洁、可执行的战术策略：
场地状态：{field_status}
要求：
1. 明确机器人的核心目标（如抢球、传球、射门、防守）；
2. 给出具体的动作指令（如移动到(3,5)位置、转向敌方球门、传球给队友A）；
3. 考虑场地约束（如边界、障碍物、队友/对手位置）；
4. 语言简洁，分点说明（最多3点）。"""
        
        if self.model_name == 'chatglm':
            tokenizer, model = self.model
            response = model.generate(tokenizer, prompt, max_length=max_tokens, temperature=0.3)
            return response[0]
        elif self.model_name == 'gpt':
            response = self.model.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        elif self.model_name == 'llama':
            tokenizer, model = self.model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def parse_natural_language_command(self, command, robot_state):
        """
        解析自然语言指令为机器人可执行的动作参数
        :param command: 自然语言指令（如"去中场抢足球"）
        :param robot_state: 机器人当前状态（位置、姿态、电量等）
        :return: 动作参数字典（如{'action': 'move', 'target_pos': (5,5), 'speed': 0.3}）
        """
        prompt = f"""你是一个人形机器人足球赛的指令解析器，将自然语言指令转换为机器人可执行的动作参数。
机器人当前状态：{robot_state}
自然语言指令：{command}
输出格式要求：
返回JSON格式，包含以下字段（根据指令类型选择必要字段）：
- action: 动作类型（move/turn/kick/pass/defend）
- target_pos: 目标位置（x,y坐标，仅move动作需要）
- target_angle: 目标角度（弧度，仅turn动作需要）
- target_teammate: 目标队友ID（仅pass动作需要）
- speed: 运动速度（0-1之间，仅move动作需要）
注意：坐标基于机器人足球场地（x范围0-10，y范围0-10，球门在(0,5)和(10,5)）"""
        
        if self.model_name == 'chatglm':
            tokenizer, model = self.model
            response = model.generate(tokenizer, prompt, max_length=300, temperature=0.1)
            return json.loads(response[0].split('\n')[-1])
        elif self.model_name == 'gpt':
            response = self.model.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return json.loads(response.choices[0].message.content.split('\n')[-1])
        elif self.model_name == 'llama':
            tokenizer, model = self.model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
            return json.loads(tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n')[-1])

# LLM应用示例
def example_llm_football_application():
    # 1. 初始化LLM客户端（以本地ChatGLM为例）
    llm_client = FootballLLMClient(
        model_name='chatglm',
        local_model_path='./chatglm-6b'  # 本地模型路径
    )
    
    # 2. 示例1：基于场地状态生成战术策略
    field_status = """当前场地状态：
    - 机器人A位置(2,3)，面向敌方球门方向；
    - 足球位置(4,5)，距离机器人A 2.2米；
    - 队友B位置(6,4)，无对手防守；
    - 敌方机器人C位置(3,6)，正朝向足球移动；
    - 我方球门在(0,5)，敌方球门在(10,5)。"""
    strategy = llm_client.generate_strategy(field_status)
    print("=== 生成的战术策略 ===")
    print(strategy)
    
    # 3. 示例2：解析自然语言指令为动作参数
    natural_command = "快速移动到足球位置，抢球后传球给队友B"
    robot_state = {"position": (2,3), "orientation": 0.0, "battery": 80, "is_holding_ball": False}
    action_params = llm_client.parse_natural_language_command(natural_command, robot_state)
    print("\n=== 解析后的动作参数 ===")
    print(json.dumps(action_params, indent=2))
    
    # 4. 示例3：多轮对话调整策略
    follow_up_command = "如果抢不到球，该怎么办？"
    multi_turn_prompt = f"""之前的场地状态：{field_status}
之前的策略：{strategy}
现在的追问：{follow_up_command}
要求：给出替代策略，简洁明了（1-2点）"""
    if llm_client.model_name == 'chatglm':
        tokenizer, model = llm_client.model
        follow_up_response = model.generate(tokenizer, multi_turn_prompt, max_length=200, temperature=0.3)[0]
    else:
        # 其他模型的多轮对话逻辑
        follow_up_response = llm_client.model.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": field_status},
                {"role": "assistant", "content": strategy},
                {"role": "user", "content": follow_up_command}
            ],
            temperature=0.3,
            max_tokens=200
        ).choices[0].message.content
    print("\n=== 追问后的替代策略 ===")
    print(follow_up_response)
```

### 4. 强化学习基础 (Chapter4/Chapter4_1/rlBasics.py)

实现了基于Q-learning的迷宫导航智能体，包括：
- 迷宫环境建模
- Q-learning算法实现
- 智能体训练与评估
- 路径可视化与动画展示

```python
# 强化学习示例
env = MazeEnvironment(size=5)
agent = QLearningAgent(None, action_size=4)

# 训练智能体
for episode in range(episodes):
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        if done:
            break
```

## 环境依赖

- Python 3.8+
- NumPy
- OpenCV
- Matplotlib
- ONNX Runtime
- PyYAML
- PyTorch 1.10+（第三章深度学习模型核心依赖）
- Transformers（Hugging Face，用于ViT、LLM加载）
- Scikit-learn（数据分割、预处理）
- OpenAI/ChatGLM等LLM相关依赖（根据使用的模型安装）

可通过以下命令安装基础依赖：
```bash
pip install numpy opencv-python matplotlib onnxruntime pyyaml torch torchvision transformers scikit-learn
```

LLM额外依赖（根据模型选择安装）：
```bash
# 远程调用GPT需安装
pip install openai
# 本地部署ChatGLM需安装
pip install sentencepiece accelerate
# 本地部署LLaMA需安装（参考Hugging Face官方指南）
pip install sentencepiece fairscale
```

## 使用方法

各章节代码可独立运行，例如：

1. 运行自注意力机制示例：
```bash
python code/Chapter3/Chapter3_1/self_attention.py
```

2. 运行LSTM时序预测示例：
```bash
python code/Chapter3/Chapter3_2/RNN_example.py
```

3. 运行ViT场地场景分类示例：
```bash
python code/Chapter3/Chapter3_3/ViT_Example.py
```

4. 运行LLM战术生成示例（需配置模型路径或API密钥）：
```bash
python code/Chapter3/Chapter3_4/LLM_example.py
```

5. 运行强化学习示例：
```bash
python code/Chapter4/Chapter4_1/rlBasics.py
```

## 注意事项

- 部分代码需要预训练模型文件：
  - ViT示例需下载预训练权重（可通过Hugging Face `transformers`库自动下载或手动放置到指定路径）
  - LLM示例需配置本地模型路径（如ChatGLM-6B、LLaMA-7B）或有效的API密钥（如GPT）
- 深度学习模型（ViT、LLM）训练/推理需要GPU支持（建议NVIDIA GPU，显存≥8GB），无GPU时需修改代码使用CPU（速度较慢）
- 可视化窗口可能需要手动关闭以继续程序执行
- LLM的prompt工程可根据实际需求调整，以获得更精准的输出结果
- 时序数据相关代码（RNN、自注意力）的序列长度、嵌入维度等参数可根据数据集调整

## 关于教材

《具身智能导论——以人形机器人足球赛为视角》以机器人足球比赛为背景，系统介绍具身智能的基本概念、核心技术和应用方法。通过理论与实践相结合的方式，帮助读者掌握感知、决策、控制等关键技术，其中第三章重点讲解注意力机制、序列模型、视觉Transformer和大语言模型在具身场景中的应用，是实现高级智能决策的核心基础。

---

希望本代码库能为您的学习提供帮助！如有问题或建议，请联系教材编委会。
