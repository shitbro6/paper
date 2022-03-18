import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2



batch_size = 4
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor (),
    transforms.Normalize((0.4915, 0.4823, 0.4468,), (1.0, 1.0, 1.0)),
])

train_dataset = datasets.CIFAR10(root='../data/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

print("训练集长度",len(train_dataset))
print("测试集长度",len(test_dataset))
# 模型类
from torch import optim



# 模型类
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.Main = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # ceil_mode=true代表向上取整

            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),

            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Inception(480, 192, 96, 208, 16, 48, 64)
        )

        self.Main2 = nn.Sequential(
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
        )

        self.Main3 = nn.Sequential(
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            # 辅助分类器
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),

        )

        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)
        # 初始化权重
        self._initialize_weights()

    # 前向传播
    def forward(self, x):
        x = self.Main(x)
        aux1 = self.aux1(x)
        x = self.Main2(x)
        aux2 = self.aux2(x)
        x = self.Main3(x)

        return x, aux2, aux1


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用凯明初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 将偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                # 权重均值是0， 标准差是0.01
                nn.init.normal_(m.weight, 0, 0.01)
                # 偏置是0
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)  # 第一个分支
        self.branch2 = nn.Sequential(  # 第二个分支
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(  # 第三个分支
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.IA1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            BasicConv2d(in_channels, 128, kernel_size=1),
            nn.Flatten(),
            nn.Dropout2d(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.IA1(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.BC1 = nn.Sequential(  # 第四个分支
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.BC1(x)
        return x





model = GoogLeNet().cuda()
# 损失函数
criterion = torch.nn.CrossEntropyLoss().cuda()
# 优化器
optimizer = optim.Adam(model.parameters(),lr=0.01)


def train(epoch):
    runing_loss = 0.0
    i = 1
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.cuda(), y.cuda()
        i +=1
        if i % 10 == 0:
            print("运行中，当前运行次数:",i)
        # 清零 正向传播  损失函数  反向传播 更新
        optimizer.zero_grad()
        # y_pre = model(x)
        # loss = criterion(y_pre, y)

        log,aux1,aux2 = model(x)
        loss0 = criterion(log, y)
        loss1 = criterion(aux1, y)
        loss2 = criterion(aux2, y)
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3


        loss.backward()
        optimizer.step()
        runing_loss += loss.item()
    # 每轮训练一共训练1W个样本，这里的runing_loss是1W个样本的总损失值，要看每一个样本的平均损失值， 记得除10000

    print("这是第 %d轮训练，当前损失值 %.5f" % (epoch + 1, runing_loss / 782))

    return runing_loss / 782

def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x, y = x.cuda(), y.cuda()
            pre_y = model(x)
            # 这里拿到的预测值 每一行都对应10个分类，这10个分类都有对应的概率，
            # 我们要拿到最大的那个概率和其对应的下标。

            j, pre_y = torch.max(pre_y.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度

            total += y.size(0)  # 统计方向0上的元素个数 即样本个数

            correct += (pre_y == y).sum().item()  # 张量之间的比较运算
    print("第%d轮测试结束，当前正确率:%d %%" % (epoch + 1, correct / total * 100))
    return correct / total * 100
if __name__ == '__main__':
    plt_epoch = []
    loss_ll = []
    corr = []
    for epoch in range(1):
        plt_epoch.append(epoch+1) # 方便绘图
        loss_ll.append(train(epoch)) # 记录每一次的训练损失值 方便绘图
        corr.append(test(epoch)) # 记录每一次的正确率

    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("训练模型")
    plt.plot(plt_epoch,loss_ll)
    plt.xlabel("循环次数")
    plt.ylabel("损失值loss")


    plt.subplot(1,2,2)
    plt.title("测试模型")
    plt.plot(plt_epoch,corr)
    plt.xlabel("循环次数")
    plt.ylabel("正确率")
    plt.show()





