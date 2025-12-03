import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ANNModel(nn.Module):
    """PyTorch神经网络模型"""

    def __init__(self, input_size, architecture='simple', dropout_rate=0.5):
        super(ANNModel, self).__init__()
        self.layers = nn.ModuleList()

        if architecture == 'simple':
            layers_config = [input_size, 12, 24, 1]
        elif architecture == 'deep':
            layers_config = [input_size, 12, 24, 48, 96, 192, 1]
        elif architecture == 'deep_with_dropout':
            layers_config = [input_size, 12, 24, 48, 96, 192, 1]
        else:
            layers_config = [input_size, 12, 24, 1]

        # 构建网络层
        for i in range(len(layers_config) - 1):
            self.layers.append(nn.Linear(layers_config[i], layers_config[i + 1]))

        self.architecture = architecture
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # 如果不是最后一层，添加激活函数
            if i < len(self.layers) - 1:
                x = self.activation(x)
                # 添加dropout（如果是deep_with_dropout架构且不是第一层）
                if (self.architecture == 'deep_with_dropout' and
                        i > 0 and i < len(self.layers) - 2):
                    x = self.dropout(x)
            else:
                x = self.output_activation(x)

        return x


class BankCustomerAnalysis:
    def __init__(self, data_path, device=None):
        """初始化分析类"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        self.df_bank = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def explore_data(self):
        """数据探索和可视化"""
        print("数据基本信息:")
        print(self.df_bank.info())
        print("\n数据前5行:")
        print(self.df_bank.head())

        # 显示不同特征的分布情况
        features = ['City', 'Gender', 'Age', 'Tenure', 'ProductsNo', 'HasCard', 'ActiveMember', 'Exited']
        fig, axes = plt.subplots(4, 2, figsize=(15, 15))
        axes = axes.ravel()

        for i, feature in enumerate(features):
            sns.countplot(x=feature, data=self.df_bank, ax=axes[i])
            axes[i].set_title(f"No. of customers - {feature}")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def preprocess_data(self):
        """数据预处理"""
        # 复制数据避免修改原始数据
        df_processed = self.df_bank.copy()

        # 二元类别文本数字化
        df_processed['Gender'] = df_processed['Gender'].map({"Female": 0, "Male": 1})
        print("Gender unique values:", df_processed['Gender'].unique())

        # 多元类别转换成哑变量
        d_city = pd.get_dummies(df_processed['City'], prefix="City")
        df_processed = pd.concat([df_processed, d_city], axis=1)

        # 构建特征和标签集合
        y = df_processed['Exited']
        X = df_processed.drop(['Name', 'Exited', 'City'], axis=1)

        # 拆分数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        return X, y

    def scale_features(self):
        """特征缩放"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled

    def create_dataloaders(self, batch_size=64):
        """创建PyTorch数据加载器"""
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(self.X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(self.X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train.values).reshape(-1, 1).to(self.device)
        y_test_tensor = torch.FloatTensor(self.y_test.values).reshape(-1, 1).to(self.device)

        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_logistic_regression(self, scaled=True):
        """训练逻辑回归模型"""
        if scaled:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train
            X_test = self.X_test

        lr = LogisticRegression(max_iter=10000)
        lr.fit(X_train, self.y_train)
        accuracy = lr.score(X_test, self.y_test) * 100
        print(f"逻辑回归测试集准确率 {accuracy:.2f}%")
        return lr

    def train_ann_model(self, model, train_loader, test_loader, epochs=30, optimizer_name='adam', lr=0.001):
        """训练PyTorch ANN模型"""
        criterion = nn.BCELoss()

        # 选择优化器
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # 记录训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        model.train()

        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # 验证阶段
            val_loss, val_acc = self.evaluate_model_pytorch(model, test_loader, criterion)

            # 记录历史
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {history["train_loss"][-1]:.4f}, '
                      f'Train Acc: {history["train_acc"][-1]:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}')

        return history, model

    def evaluate_model_pytorch(self, model, test_loader, criterion):
        """评估PyTorch模型"""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        model.train()
        return test_loss / len(test_loader), correct / total

    def predict_pytorch(self, model, test_loader):
        """使用PyTorch模型进行预测"""
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        model.train()
        return np.array(all_predictions), np.array(all_targets)

    def evaluate_model(self, y_test, y_pred):
        """评估模型并显示结果"""
        # 显示分类报告和混淆矩阵
        self.show_report(y_test, y_pred)
        self.show_matrix(y_test, y_pred)

    def show_history(self, history):
        """显示训练过程中的学习曲线"""
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'bo', label='Training loss')
        plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'bo', label='Training acc')
        plt.plot(epochs, history['val_acc'], 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def show_report(self, y_test, y_pred):
        """显示分类报告"""
        print("分类报告:")
        print(classification_report(y_test, y_pred, labels=[0, 1]))

    def show_matrix(self, y_test, y_pred):
        """显示混淆矩阵"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show(block=False)
        plt.pause(0.1)


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    # 初始化分析器
    analyzer = BankCustomerAnalysis("数据集/BankCustomer.csv")

    # 数据探索
    analyzer.explore_data()

    # 数据预处理
    X, y = analyzer.preprocess_data()

    # 特征缩放
    analyzer.scale_features()

    print("=" * 50)
    print("1. 逻辑回归模型")
    print("=" * 50)
    # 训练逻辑回归模型（未缩放特征）
    print("未缩放特征:")
    lr_unscaled = analyzer.train_logistic_regression(scaled=False)

    # 训练逻辑回归模型（缩放特征）
    print("缩放特征后:")
    lr_scaled = analyzer.train_logistic_regression(scaled=True)

    # 创建数据加载器
    train_loader, test_loader = analyzer.create_dataloaders(batch_size=64)
    input_size = analyzer.X_train_scaled.shape[1]

    print("\n" + "=" * 50)
    print("2. 简单神经网络")
    print("=" * 50)
    # 简单神经网络
    ann_simple = ANNModel(input_size, architecture='simple').to(analyzer.device)
    history_simple, model_simple = analyzer.train_ann_model(
        ann_simple, train_loader, test_loader, optimizer_name='adam')

    y_pred_simple, y_test_simple = analyzer.predict_pytorch(model_simple, test_loader)
    analyzer.show_history(history_simple)
    analyzer.evaluate_model(y_test_simple, y_pred_simple)

    print("\n" + "=" * 50)
    print("3. 深层神经网络")
    print("=" * 50)
    # 深层神经网络
    ann_deep = ANNModel(input_size, architecture='deep').to(analyzer.device)
    history_deep, model_deep = analyzer.train_ann_model(
        ann_deep, train_loader, test_loader, optimizer_name='rmsprop')

    y_pred_deep, y_test_deep = analyzer.predict_pytorch(model_deep, test_loader)
    analyzer.show_history(history_deep)
    analyzer.evaluate_model(y_test_deep, y_pred_deep)

    print("\n" + "=" * 50)
    print("4. 带Dropout的深层神经网络")
    print("=" * 50)
    # 带Dropout的深层神经网络
    ann_dropout = ANNModel(input_size, architecture='deep_with_dropout', dropout_rate=0.5).to(analyzer.device)
    history_dropout, model_dropout = analyzer.train_ann_model(
        ann_dropout, train_loader, test_loader, optimizer_name='adam')

    y_pred_dropout, y_test_dropout = analyzer.predict_pytorch(model_dropout, test_loader)
    analyzer.show_history(history_dropout)
    analyzer.evaluate_model(y_test_dropout, y_pred_dropout)

    # 保持图形窗口打开
    print("\n所有图形已显示，关闭图形窗口将结束程序...")
    plt.show()


if __name__ == "__main__":
    main()