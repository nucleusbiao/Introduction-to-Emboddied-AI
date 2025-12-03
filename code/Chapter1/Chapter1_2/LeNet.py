import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class LeNet5(nn.Module):
    """LeNet-5模型用于MNIST手写数字识别"""

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # C1: 卷积层 1@32x32 -> 6@28x28
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            # S2: 池化层 6@28x28 -> 6@14x14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: 卷积层 6@14x14 -> 16@10x10
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            # S4: 池化层 16@10x10 -> 16@5x5
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C5: 卷积层 16@5x5 -> 120@1x1
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        # 分类器层
        self.classifier = nn.Sequential(
            # F6: 全连接层 120 -> 84
            nn.Linear(120, 84),
            nn.Tanh(),
            # 输出层: 84 -> 10
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return x


class MNISTAnalysis:
    def __init__(self, device=None):
        """初始化MNIST分析类"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # LeNet-5需要32x32输入
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
        ])

        self.train_loader = None
        self.test_loader = None
        self.model = None

    def load_data(self, batch_size=64):
        """加载MNIST数据集"""
        # 训练集
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )

        # 测试集
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")

        return self.train_loader, self.test_loader

    def explore_data(self):
        """数据探索和可视化"""
        if self.train_loader is None:
            self.load_data()

        # 获取一个批次的数据
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)

        print(f"图像张量形状: {images.shape}")  # [batch_size, channels, height, width]
        print(f"标签形状: {labels.shape}")
        print(f"像素值范围: [{images.min():.3f}, {images.max():.3f}]")

        # 显示一些样本图像
        self.show_sample_images(images, labels)

    def show_sample_images(self, images, labels, num_samples=12):
        """显示样本图像"""
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()

        for i in range(num_samples):
            img = images[i].squeeze().numpy()  # 移除通道维度
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {labels[i].item()}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def create_model(self, num_classes=10):
        """创建LeNet-5模型"""
        self.model = LeNet5(num_classes=num_classes).to(self.device)
        print("LeNet-5模型已创建:")
        print(self.model)
        return self.model

    def train_model(self, epochs=10, learning_rate=0.001, optimizer_name='adam'):
        """训练模型"""
        if self.model is None:
            self.create_model()
        if self.train_loader is None:
            self.load_data()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()

        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

        # 记录训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        print("开始训练...")

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 验证阶段
            val_loss, val_acc = self.evaluate_model()

            # 更新学习率
            scheduler.step()

            # 记录历史
            history['train_loss'].append(train_loss / len(self.train_loader))
            history['train_acc'].append(train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {history["train_loss"][-1]:.4f}, '
                  f'Train Acc: {history["train_acc"][-1]:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

        return history

    def evaluate_model(self):
        """评估模型性能"""
        if self.model is None or self.test_loader is None:
            raise ValueError("模型或测试数据未初始化")

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()
        return test_loss / len(self.test_loader), correct / total

    def predict(self):
        """使用模型进行预测"""
        if self.model is None:
            raise ValueError("模型未初始化")

        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        self.model.train()
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)

    def show_history(self, history):
        """显示训练历史"""
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Training accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def show_confusion_matrix(self, y_true, y_pred):
        """显示混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show(block=False)
        plt.pause(0.1)

    def show_classification_report(self, y_true, y_pred):
        """显示分类报告"""
        print("分类报告:")
        print(classification_report(y_true, y_pred, digits=4))

    def test_individual_samples(self, num_samples=5):
        """测试单个样本"""
        if self.test_loader is None:
            self.load_data()

        self.model.eval()
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

        for i in range(num_samples):
            img = images[i].unsqueeze(0).to(self.device)  # 添加batch维度
            with torch.no_grad():
                output = self.model(img)
                probability = torch.softmax(output, dim=1)
                predicted = torch.argmax(output, 1)

            img_display = images[i].squeeze().numpy()
            axes[i].imshow(img_display, cmap='gray')
            axes[i].set_title(
                f'True: {labels[i]}\nPred: {predicted.item()}\nProb: {probability[0][predicted].item():.3f}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化MNIST分析器
    analyzer = MNISTAnalysis()

    print("=" * 50)
    print("1. 数据探索")
    print("=" * 50)
    # 数据探索
    analyzer.explore_data()

    print("\n" + "=" * 50)
    print("2. 创建LeNet-5模型")
    print("=" * 50)
    # 创建模型
    analyzer.create_model()

    print("\n" + "=" * 50)
    print("3. 训练模型")
    print("=" * 50)
    # 训练模型
    history = analyzer.train_model(epochs=15, learning_rate=0.001, optimizer_name='adam')

    print("\n" + "=" * 50)
    print("4. 模型评估")
    print("=" * 50)
    # 最终评估
    final_loss, final_acc = analyzer.evaluate_model()
    print(f"最终测试集准确率: {final_acc:.4f}")
    print(f"最终测试集损失: {final_loss:.4f}")

    # 显示训练历史
    analyzer.show_history(history)

    # 进行预测
    y_pred, y_true, probabilities = analyzer.predict()

    # 显示混淆矩阵
    analyzer.show_confusion_matrix(y_true, y_pred)

    # 显示分类报告
    analyzer.show_classification_report(y_true, y_pred)

    print("\n" + "=" * 50)
    print("5. 单个样本测试")
    print("=" * 50)
    # 测试单个样本
    analyzer.test_individual_samples(num_samples=8)

    # 保持图形窗口打开
    print("\n所有图形已显示，关闭图形窗口将结束程序...")
    plt.show()


if __name__ == "__main__":
    main()