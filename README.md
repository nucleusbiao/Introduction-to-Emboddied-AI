# 《具身智能导论——以人形机器人足球赛为视角》案例代码库

欢迎使用《具身智能导论——以人形机器人足球赛为视角》的配套案例代码库。本代码库严格遵循教材章节结构，提供了书中所有核心算法的可运行实现与实验案例，旨在帮助读者将理论知识与工程实践相结合，深入理解具身智能在人形机器人足球场景中的应用逻辑与技术细节。

## 代码结构

```
Introduction-to-Emboddied-AI/
├── code/
│   ├── Chapter1/           # 第一章：基础知识
│   │   ├── Chapter1_1/
│   │   │   ├── 3_classification.py  # 分类算法示例（三分类任务）
│   │   │   └── bankCustomer.py       # 银行客户相关分类/预测任务示例
│   │   ├── Chapter1_2/
│   │   │   ├── LeNet.py              # LeNet卷积神经网络实现（图像分类基础）
│   │   │   └── yolov11.py            # YOLOv11目标检测基础实现（简化版）
│   │   └── Chapter1_3/
│   │       └── main.py               # 第一章综合应用示例（多算法整合演示）
│   ├── Chapter2/           # 第二章：感知与定位
│   │   ├── Chapter2_1/
│   │   │   ├── task1.py              # 视觉感知基础任务（如图像预处理、特征提取）
│   │   │   └── task2.py              # 足球检测基础实现（基于ONNX模型）
│   │   ├── Chapter2_2/
│   │   │   └── cam_calibrate.py      # 相机标定工具（内参计算、畸变校正）
│   │   └── Chapter2_3/
│   │       ├── robot_localization.py  # 机器人定位核心模块（场地标记+坐标转换）
│   │       ├── camPosePF.py           # 基于粒子滤波的相机位姿估计
│   │       └── particleFilterSim.py   # 粒子滤波算法仿真（定位算法验证工具）
│   ├── Chapter3/           # 第三章：注意力机制与大语言模型应用
│   │   ├── Chapter3_1/
│   │   │   └── self_attention.py      # 自注意力机制基础实现
│   │   ├── Chapter3_2/
│   │   │   └── RNN_example.py         # 循环神经网络应用示例（时序数据建模）
│   │   ├── Chapter3_3/
│   │   │   └── ViT_Example.py         # 视觉Transformer（ViT）应用示例
│   │   └── Chapter3_4/
│   │       └── LLM_example.py         # 大语言模型（LLM）接口与应用示例
│   └── Chapter4/           # 第四章：强化学习基础与进阶
│       ├── Chapter4_1/
│       │   └── rlBasics.py           # 强化学习基础（Q-learning迷宫导航）
│       ├── Chapter4_2/
│       │   └── reinforcement.py      # 强化学习进阶应用（通用场景扩展）
│       ├── Chapter4_3/
│       │   └── DQN.py                # 深度Q网络（DQN）实现（值函数近似）
│       └── Chapter4_4/
│           └── A3C_PPO_SAC.py        # 高级强化学习算法集合（A3C、PPO、SAC）
├── README.md                          # 代码库说明文档
└── requirements.txt                   # 依赖库清单文件
```

## 各章节功能说明

### 第一章：基础知识
本章聚焦深度学习基础理论与工程实现，覆盖传统神经网络、卷积神经网络、目标检测入门等核心内容，为后续复杂模块打下基础。

- **Chapter1_1**
  - `3_classification.py`：深度学习入门示例，实现基于全连接神经网络的三分类任务，包含数据集生成、模型搭建、训练与评估全流程。
  - `bankCustomer.py`：基于真实场景的分类/预测任务，以银行客户数据为载体，演示数据预处理、特征工程、模型优化的实践方法。
- **Chapter1_2**
  - `LeNet.py`：经典LeNet-5卷积神经网络实现，针对灰度图像分类设计，帮助理解卷积、池化等核心操作的原理与应用。
  - `yolov11.py`：YOLOv11目标检测算法简化实现，涵盖目标检测的核心逻辑（候选框生成、分类与回归、非极大值抑制），适配机器人视觉检测场景。
- **Chapter1_3**
  - `main.py`：第一章综合应用示例，整合前两小节的算法，实现"数据处理→特征提取→模型推理→结果可视化"的完整流程演示。

### 第二章：感知与定位
本章围绕机器人足球场景的感知需求，实现从图像采集、相机标定到机器人定位的全链路技术，是机器人自主运动的基础。

- **Chapter2_1**
  - `task1.py`：视觉感知基础任务，包含图像读取、灰度化、滤波、边缘检测等预处理操作，为后续目标检测提供高质量输入。
  - `task2.py`：基于ONNX预训练模型的足球检测实现，涵盖模型加载、图像预处理、推理计算、检测框绘制等核心步骤。
- **Chapter2_2**
  - `cam_calibrate.py`：相机标定工具，通过棋盘格标定板计算相机内参矩阵与畸变系数，实现图像畸变校正，提升定位精度。
- **Chapter2_3**
  - `robot_localization.py`：机器人定位核心模块，结合场地标记检测与坐标转换算法，实现图像坐标到世界坐标的映射。
  - `camPosePF.py`：基于粒子滤波的相机位姿估计，通过粒子群采样、权重更新、重采样等步骤，实现鲁棒的位姿预测。
  - `particleFilterSim.py`：粒子滤波算法仿真工具，可独立验证粒子滤波的收敛性、鲁棒性，辅助调试定位参数。

### 第三章：注意力机制与大语言模型应用
本章聚焦前沿深度学习技术，将注意力机制、序列模型、视觉Transformer与大语言模型融入具身智能场景，提升机器人的理解与决策能力。

- **Chapter3_1**
  - `self_attention.py`：缩放点积注意力、多头注意力机制的基础实现，支持位置编码与注意力权重可视化，适配时序数据（如动作序列、状态序列）的关键信息提取。
- **Chapter3_2**
  - `RNN_example.py`：基础RNN、LSTM、GRU模型实现，针对时序数据建模设计，可用于足球位置预测、队友/对手运动轨迹估计等场景。
- **Chapter3_3**
  - `ViT_Example.py`：视觉Transformer（ViT）完整实现，包含Patch Embedding、Transformer Encoder、分类头等模块，适用于场地场景分类、机器人动作状态识别等视觉任务。
- **Chapter3_4**
  - `LLM_example.py`：主流大语言模型（ChatGLM、GPT、LLaMA）的调用封装，支持API远程调用与本地部署，实现自然语言指令解析、比赛战术生成、多轮对话交互等智能决策功能。

### 第四章：强化学习基础与进阶
本章从基础到进阶，系统覆盖强化学习核心算法，适配机器人足球场景的决策与控制需求，实现智能体的自主学习与优化。

- **Chapter4_1**
  - `rlBasics.py`：强化学习入门示例，基于Q-learning算法实现迷宫导航智能体，帮助理解智能体与环境交互、奖励机制设计、策略优化的基本逻辑。
- **Chapter4_2**
  - `reinforcement.py`：强化学习进阶应用，扩展基础算法到更复杂的场景（如多目标任务、动态环境），演示强化学习的工程化适配方法。
- **Chapter4_3**
  - `DQN.py`：深度Q网络（DQN）实现，通过神经网络近似值函数，解决高维状态空间下的决策问题，适配机器人运动控制、战术选择等场景。
- **Chapter4_4**
  - `A3C_PPO_SAC.py`：高级强化学习算法集合，包含异步优势演员-评论员（A3C）、近端策略优化（PPO）、软演员-评论员（SAC），适用于复杂动态场景下的高效决策与控制。

## 环境依赖

### 基础依赖（必装）
- Python 3.8+
- numpy、matplotlib、opencv-python、pyyaml、onnxruntime
- scikit-learn（数据处理、评估与数据集分割）

### 深度学习依赖（核心）
- PyTorch 1.10+（模型构建、训练与推理）
- torchvision（视觉数据处理、数据集加载）
- transformers（Hugging Face，预训练ViT、LLM模型加载）

### 强化学习专项依赖
- gym（可选，部分强化学习环境建模）
- tensorboard（可选，训练过程可视化）

### LLM专项依赖（按需安装）
- openai（远程调用GPT系列模型）
- sentencepiece（LLM文本分词处理）
- accelerate（本地LLM部署加速）
- fairscale（大模型分布式推理与训练）

## 安装指南

1. 克隆代码库到本地
```bash
git clone https://github.com/xxx/Introduction-to-Emboddied-AI.git
cd Introduction-to-Emboddied-AI
```

2. 安装基础依赖
```bash
pip install -r requirements.txt
```

3. 安装专项依赖（按需选择）
```bash
# 强化学习可视化（可选）
pip install tensorboard gym

# 远程调用GPT
pip install openai

# 本地部署ChatGLM
pip install sentencepiece accelerate

# 本地部署LLaMA
pip install sentencepiece fairscale
```

4. 无GPU环境适配（可选）
若无NVIDIA GPU，安装CPU版本PyTorch（推理速度较慢，大模型可能无法运行）：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 使用方法

各章节代码独立可运行，直接通过Python命令执行对应文件即可：

```bash
# 第一章示例：运行LeNet卷积神经网络
python code/Chapter1/Chapter1_2/LeNet.py

# 第二章示例：运行相机标定
python code/Chapter2/Chapter2_2/cam_calibrate.py

# 第三章示例：运行ViT场景分类
python code/Chapter3/Chapter3_3/ViT_Example.py

# 第四章示例：运行PPO算法
python code/Chapter4/Chapter4_4/A3C_PPO_SAC.py
```

## 注意事项

1. 预训练模型要求：
   - Chapter1的yolov11.py、Chapter2的task2.py需对应ONNX预训练模型；
   - Chapter3的ViT_Example.py、LLM_example.py需预训练权重（可通过Hugging Face自动下载或参考教材获取）；
   - 本地部署LLM建议配备显存≥8GB的NVIDIA GPU，否则需启用CPU推理（速度极慢，仅用于学习演示）。

2. 数据与配置：
   - 部分代码（如bankCustomer.py、cam_calibrate.py）需配套数据集或标定板图像，参考教材说明获取；
   - 可通过修改代码中的参数（如学习率、迭代次数、模型维度）适配不同硬件环境与任务需求。

3. 运行说明：
   - 可视化窗口（如matplotlib、OpenCV绘制的结果）需手动关闭以继续程序执行；
   - 强化学习算法（如DQN、PPO）训练时间较长，可根据硬件配置调整训练轮数与批次大小；
   - LLM_example.py需提前配置API密钥（远程调用）或本地模型路径（本地部署），否则无法正常运行。

4. 版本兼容性：
   - 建议严格按照`requirements.txt`指定版本安装依赖，避免因版本不兼容导致报错；
   - 若遇到PyTorch相关问题，优先升级到1.18+版本（适配最新transformers库功能）。

## 关于教材

《具身智能导论——以人形机器人足球赛为视角》以人形机器人足球赛为核心应用场景，系统讲解具身智能的基本概念、核心技术与工程实践。全书涵盖基础知识、感知定位、注意力机制与大语言模型、强化学习四大模块，通过"理论+案例"的模式，帮助读者快速掌握具身智能的关键技术与应用方法。

## 反馈与支持

若在使用代码过程中遇到问题或有优化建议，欢迎联系教材编委会或通过代码库Issue提交反馈。

---

祝学习愉快！
