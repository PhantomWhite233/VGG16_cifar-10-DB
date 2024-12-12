import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将 CIFAR-10 的 32x32 图像调整为 VGG16 输入的 224x224 图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 加载预训练的 VGG16 模型
model = models.vgg16(pretrained=True)

# 冻结卷积层的参数，只训练最后的全连接层
for param in model.parameters():
    param.requires_grad = False

# 修改最后的全连接层，适应 CIFAR-10 的 10 个类别
model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 有 10 个类别

# 将模型移到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # 只优化全连接层的参数

# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 训练过程
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", delay=0.1):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确度
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # 训练准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # 输出训练结果
        print(f"Epoch [{epoch + 1}/{epochs}]:")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.2f}%")

# 测试模型
def test_model(model, test_loader, criterion):
    model.eval()  # 切换到评估模式
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss_test += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    # 测试集损失与准确率
    test_loss = running_loss_test / len(test_loader)
    test_accuracy = 100 * correct_test / total_test

    # 输出测试结果
    print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.2f}%")


# 训练并测试模型
EPOCHS = 10
train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)
test_model(model, test_loader, criterion)
