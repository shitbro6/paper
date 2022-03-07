import torch
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


batch_size = 1
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor (),
    transforms.Normalize((0.485, 0.456, 0.406), ( 0.229, 0.224, 0.225)),
])

train_dataset = datasets.FashionMNIST(root='../data/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.FashionMNIST(root='../data/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

print("训练集长度",len(train_dataset))
print("测试集长度",len(test_dataset))

# 模型类设计

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.mode1 = nn.Sequential(
            #1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            #2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            #4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2,stride=2),

            #5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            #6
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            #7
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(stride=2, kernel_size=2),

            #8
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #9
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #10
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(stride=2, kernel_size=2),

            #11
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #12
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #13
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # 线性层
            nn.Flatten(),
            #14
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5,),
            #15
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5,),
            #16
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(True),
        )
        self._initialize_weights()

    def forward(self, input):

        x = self.mode1(input)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

model = VGG16().cuda()
# 损失函数
criterion = torch.nn.CrossEntropyLoss().cuda()
# 优化器A
optimizer = optim.SGD(model.parameters(),lr=0.01,weight_decay=0.0005,momentum=0.9)


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
        y_pre = model(x)
        loss = criterion(y_pre, y)
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
    for epoch in range(20):
        plt_epoch.append(epoch+1) # 方便绘图
        loss_ll.append(train(epoch)) # 记录每一次的训练损失值 方便绘图
        corr.append(test(epoch)) # 记录每一次的正确率
# 可视化
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









