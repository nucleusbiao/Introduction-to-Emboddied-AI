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

### 2. 目标检测与定位 (Chapter2/Chapter2_1 ~ Chapter2_3)

#### 2.1 足球检测 (Chapter2/Chapter2_1/task2.py)
实现了基于ONNX模型的足球检测功能，包括：
- 模型加载与初始化
- 图像预处理与推理
- 检测结果后处理
- 检测结果可视化


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


### 3. 注意力机制与大语言模型应用 (Chapter3/Chapter3_1 ~ Chapter3_4)
第三章聚焦具身智能中的核心表征学习技术，通过注意力机制、序列模型、视觉Transformer和大语言模型的实现与应用，解决机器人足球场景中的感知理解、序列决策等问题。

#### 3.1 自注意力机制基础 (Chapter3/Chapter3_1/self_attention.py)
实现自注意力机制的核心原理与简化版实现，适配机器人足球场景中的序列信息处理（如动作序列、感知时序数据），包括：
- 缩放点积注意力（Scaled Dot-Product Attention）核心计算
- 多头注意力（Multi-Head Attention）并行特征提取
- 位置编码（Positional Encoding）时序/空间信息注入
- 注意力权重可视化与解释
- 适配机器人动作序列、场地状态序列的处理接口



#### 3.2 循环神经网络应用 (Chapter3/Chapter3_2/RNN_example.py)
实现循环神经网络（RNN）及其变体（LSTM、GRU），用于机器人足球场景中的时序数据建模（如状态预测、动作序列生成），包括：
- 基础RNN、LSTM、GRU模型构建
- 场地状态时序预测（如足球位置、队友/对手运动轨迹预测）
- 动作序列生成（基于历史动作预测下一阶段动作）
- 时序数据预处理与序列长度适配
- 模型训练与预测结果可视化


#### 3.3 视觉Transformer（ViT）应用 (Chapter3/Chapter3_3/ViT_Example.py)
实现视觉Transformer（Vision Transformer, ViT）模型，用于机器人足球场景中的图像理解任务（如场地场景分类、目标检测辅助、动作状态识别），包括：
- ViT核心模块实现（Patch Embedding、Transformer Encoder、分类头）
- 场地场景分类（如进攻区、防守区、中场区识别）
- 目标状态识别（如足球是否在控制中、机器人动作状态分类）
- 预训练权重加载与微调适配
- 图像输入预处理与注意力热力图可视化



#### 3.4 大语言模型（LLM）接口与应用 (Chapter3/Chapter3_4/LLM_example.py)
实现大语言模型（LLM）的调用接口与具身智能场景应用，聚焦机器人足球中的自然语言交互、策略生成、指令理解等任务，包括：
- 主流LLM（如GPT、LLaMA、ChatGLM）的API/本地调用封装
- 机器人指令理解（自然语言→动作指令映射）
- 比赛策略生成（基于场地状态描述生成战术建议）
- 多轮对话交互（与机器人进行比赛相关的自然语言沟通）
- prompt工程优化（适配具身场景的指令设计）



### 4. 强化学习基础 (Chapter4/Chapter4_1/rlBasics.py)

实现了基于Q-learning的迷宫导航智能体，包括：
- 迷宫环境建模
- Q-learning算法实现
- 智能体训练与评估
- 路径可视化与动画展示


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
