# 《具身智能导论——以人形人形机器人足球赛为视角》案例代码库

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

### 2. 目标检测 (Chapter2/Chapter2_1/task2.py)

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

### 3. 机器人定位 (Chapter2/Chapter2_3/)

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

- Python 3.7+
- NumPy
- OpenCV
- Matplotlib
- ONNX Runtime
- PyYAML

可通过以下命令安装主要依赖：
```bash
pip install numpy opencv-python matplotlib onnxruntime pyyaml
```

## 使用方法

各章节代码可独立运行，例如：

1. 运行分类算法示例：
```bash
python code/Chapter1/Chapter1_1/3_classification.py
```

2. 运行强化学习示例：
```bash
python code/Chapter4/Chapter4_1/rlBasics.py
```

## 注意事项

- 部分代码需要预训练模型文件，请参考教材中的说明获取
- 可视化窗口可能需要手动关闭以继续程序执行
- 不同硬件配置可能需要调整参数以获得最佳性能

## 关于教材

《具身智能导论——以人形机器人足球赛为视角》以机器人足球比赛为背景，系统介绍具身智能的基本概念、核心技术和应用方法。通过理论与实践相结合的方式，帮助读者掌握感知、决策、控制等关键技术。

---

希望本代码库能为您的学习提供帮助！如有问题或建议，请联系教材编委会。
