import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings('ignore')  # 忽略警告


class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.01, activation='relu'):
        """
        初始化深度神经网络

        Args:
            layer_dims: 网络结构，如 [input_dim, hidden1_dim, ..., output_dim]
            learning_rate: 学习率
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.parameters = {}
        self.gradients = {}
        self.cache = {}

        # 初始化参数
        self._initialize_parameters()

    def _initialize_parameters(self):
        """使用He初始化方法初始化权重和偏置"""
        np.random.seed(1)
        L = len(self.layer_dims)

        for l in range(1, L):
            # He初始化，适合ReLU激活函数
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]) * np.sqrt(2.0 / self.layer_dims[l - 1])
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    def _activation(self, Z, derivative=False):
        """激活函数及其导数"""
        if self.activation_type == 'relu':
            if derivative:
                return (Z > 0).astype(float)
            return np.maximum(0, Z)

        elif self.activation_type == 'sigmoid':
            s = 1 / (1 + np.exp(-Z))
            if derivative:
                return s * (1 - s)
            return s

        elif self.activation_type == 'tanh':
            if derivative:
                return 1 - np.tanh(Z) ** 2
            return np.tanh(Z)

    def _softmax(self, Z):
        """Softmax函数"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # 数值稳定性
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def _forward_propagation(self, X):
        """前向传播"""
        self.cache['A0'] = X
        A_prev = X
        L = len(self.parameters) // 2  # 层数

        # 隐藏层前向传播
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b

            self.cache[f'Z{l}'] = Z
            A_prev = self._activation(Z)
            self.cache[f'A{l}'] = A_prev

        # 输出层（使用softmax）
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        Z = np.dot(W, A_prev) + b
        self.cache[f'Z{L}'] = Z
        AL = self._softmax(Z)
        self.cache[f'A{L}'] = AL

        return AL

    def _compute_cost(self, AL, Y):
        """计算交叉熵损失"""
        m = Y.shape[1]

        # 数值稳定性处理
        AL = np.clip(AL, 1e-15, 1 - 1e-15)

        # 交叉熵损失
        cost = -np.sum(Y * np.log(AL)) / m

        return cost

    def _backward_propagation(self, AL, Y):
        """反向传播计算梯度"""
        m = Y.shape[1]
        L = len(self.parameters) // 2

        # 初始化反向传播
        dAL = - (Y / AL)  # 交叉熵损失的导数

        # 输出层梯度
        dZ = AL - Y  # softmax + 交叉熵的简化梯度
        self.gradients[f'dW{L}'] = np.dot(dZ, self.cache[f'A{L - 1}'].T) / m
        self.gradients[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True) / m

        # 隐藏层反向传播
        dA_prev = np.dot(self.parameters[f'W{L}'].T, dZ)

        for l in reversed(range(1, L)):
            dZ = dA_prev * self._activation(self.cache[f'Z{l}'], derivative=True)
            self.gradients[f'dW{l}'] = np.dot(dZ, self.cache[f'A{l - 1}'].T) / m
            self.gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)

    def _update_parameters(self):
        """使用梯度下降更新参数"""
        L = len(self.parameters) // 2

        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * self.gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.gradients[f'db{l}']

    def train(self, X, Y, epochs=1000, verbose=True):
        """
        训练神经网络

        Args:
            X: 输入数据 (特征数, 样本数)
            Y: 标签 (类别数, 样本数)
            epochs: 训练轮数
            verbose: 是否打印训练过程
        """
        costs = []

        for i in range(epochs):
            # 前向传播
            AL = self._forward_propagation(X)

            # 计算损失
            cost = self._compute_cost(AL, Y)
            costs.append(cost)

            # 反向传播
            self._backward_propagation(AL, Y)

            # 更新参数
            self._update_parameters()

            if verbose and i % 100 == 0:
                print(f"Epoch {i}, Cost: {cost:.6f}")

        return costs

    def predict(self, X):
        """预测"""
        AL = self._forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

    def accuracy(self, X, Y):
        """计算准确率"""
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == true_labels)
        return accuracy


def create_better_3class_data(n_samples=900, n_features=20, n_classes=3):
    """
    创建更好的三分类数据集，使其在低维空间更容易可视化
    """
    np.random.seed(42)

    # 为每个类别创建不同的均值向量
    means = [
        np.random.randn(n_features) * 2,  # 类别0
        np.random.randn(n_features) * 2 + 5,  # 类别1
        np.random.randn(n_features) * 2 - 5  # 类别2
    ]

    # 为每个类别创建不同的协方差矩阵
    covs = []
    for i in range(n_classes):
        # 创建正定协方差矩阵
        A = np.random.randn(n_features, n_features)
        cov = np.dot(A, A.T) + np.eye(n_features) * 0.1
        covs.append(cov)

    # 生成数据
    X_list = []
    y_list = []
    samples_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        X_class = np.random.multivariate_normal(
            means[class_idx], covs[class_idx], samples_per_class
        )
        X_list.append(X_class.T)
        y_list.extend([class_idx] * samples_per_class)

    # 合并数据
    X = np.hstack(X_list)
    Y = np.eye(n_classes)[y_list].T

    return X, Y


def split_dataset(X, Y, test_size=0.2, random_state=42):
    """
    将数据集按照比例分成训练集和测试集

    Args:
        X: 输入数据 (特征数, 样本数)
        Y: 标签 (类别数, 样本数)
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        X_train, X_test, Y_train, Y_test
    """
    np.random.seed(random_state)
    m = X.shape[1]  # 样本数量

    # 随机打乱数据
    indices = np.random.permutation(m)
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    # 分割数据
    split_idx = int(m * (1 - test_size))

    X_train = X_shuffled[:, :split_idx]
    X_test = X_shuffled[:, split_idx:]
    Y_train = Y_shuffled[:, :split_idx]
    Y_test = Y_shuffled[:, split_idx:]

    return X_train, X_test, Y_train, Y_test


def plot_3d_data(X, Y, title="三分类数据三维可视化", block=False):
    """
    在三维坐标轴上显示三分类数据点

    Args:
        X: 输入数据 (特征数, 样本数)
        Y: 标签 (类别数, 样本数)
        title: 图表标题
        block: 是否阻塞显示
    """
    # 获取真实标签
    true_labels = np.argmax(Y, axis=0)

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))

    explained_variance = None
    if X.shape[0] > 3:
        # 使用前三个特征
        X_3d = X[:3, :]
        dim_labels = ['特征 1', '特征 2', '特征 3']
    else:
        X_3d = X
        dim_labels = ['特征 1', '特征 2', '特征 3']

    ax = fig.add_subplot(111, projection='3d')

    # 为每个类别设置不同的颜色和标记
    colors = ['red', 'blue', 'green']
    markers = ['o', '^', 's']
    labels = ['类别 0', '类别 1', '类别 2']

    for i in range(3):
        mask = (true_labels == i)
        ax.scatter(X_3d[0, mask], X_3d[1, mask], X_3d[2, mask],
                   c=colors[i], marker=markers[i], label=labels[i],
                   alpha=0.7, s=30, edgecolors='w', linewidth=0.5)

    ax.set_xlabel(dim_labels[0])
    ax.set_ylabel(dim_labels[1])
    ax.set_zlabel(dim_labels[2])

    # 修复这里：检查 explained_variance 是否为 None
    if explained_variance is not None:
        ax.set_title(f'{title}\n(累计解释方差: {sum(explained_variance):.2%})')
    else:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()

    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.1)  # 短暂暂停以确保图形更新

    return explained_variance


# 训练示例
def example_training():
    # 创建更好的三分类数据集
    print("生成三分类数据...")
    n_samples = 900
    n_features = 20
    n_classes = 3

    X, Y = create_better_3class_data(n_samples, n_features, n_classes)

    print("\n显示完整数据集分布 (前三个特征)...")
    plot_3d_data(X, Y, "完整数据集分布 - 前三个特征", block=False)

    # 分割数据集
    print("\n分割数据集 (训练集80%, 测试集20%)...")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, test_size=0.2)

    print(f"训练集大小: {X_train.shape[1]} 个样本")
    print(f"测试集大小: {X_test.shape[1]} 个样本")

    # 显示训练集和测试集分布
    print("\n显示训练集分布...")
    plot_3d_data(X_train, Y_train, "训练集分布 - 前三个特征", block=False)

    print("显示测试集分布...")
    plot_3d_data(X_test, Y_test, "测试集分布 - 前三个特征", block=False)

    # 创建神经网络
    layer_dims = [n_features, 64, 32, n_classes]  # 输入-隐藏层1-隐藏层2-输出
    nn = DeepNeuralNetwork(layer_dims, learning_rate=0.1, activation='relu')

    # 训练（只使用训练集）
    print("\n开始训练...")
    costs = nn.train(X_train, Y_train, epochs=1000, verbose=True)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('训练损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1)

    # 计算训练集和测试集准确率
    train_accuracy = nn.accuracy(X_train, Y_train)
    test_accuracy = nn.accuracy(X_test, Y_test)

    print(f"\n=== 模型性能评估 ===")
    print(f"训练集准确率: {train_accuracy * 100:.2f}%")
    print(f"测试集准确率: {test_accuracy * 100:.2f}%")

    # 显示预测结果对比
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)

    train_true_labels = np.argmax(Y_train, axis=0)
    test_true_labels = np.argmax(Y_test, axis=0)

    print(f"\n训练集错误分类数量: {np.sum(train_predictions != train_true_labels)} / {X_train.shape[1]}")
    print(f"测试集错误分类数量: {np.sum(test_predictions != test_true_labels)} / {X_test.shape[1]}")

    # 绘制准确率对比图
    plt.figure(figsize=(8, 6))
    categories = ['训练集', '测试集']
    accuracies = [train_accuracy * 100, test_accuracy * 100]
    colors = ['blue', 'orange']

    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7)
    plt.title('训练集 vs 测试集准确率')
    plt.ylabel('准确率 (%)')
    plt.ylim(0, 100)

    # 在柱状图上显示数值
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    plt.pause(0.1)

    # 保持图形窗口打开
    print("\n所有图形已显示，关闭图形窗口将结束程序...")
    plt.show()


if __name__ == "__main__":
    example_training()