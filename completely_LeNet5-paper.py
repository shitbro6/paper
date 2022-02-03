import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

'''
定义数据的初始化方法：
    1. 将数据转化成tensor
    2. 将数据的灰度值范围从[0, 1]转化为[-0.1, 1.175]
    3. 将数据进行尺寸变化的操作我们放在卷积层的padding操作中，这样更加方便
'''

picProcessor = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean = [0.1 / 1.275],
        std = [1.0 / 1.275]
    ),
])

'''
数据的读取和处理：
    1. 从官网下载太慢了，所以先重新指定路径，并且在mnist.py文件里把url改掉
    2. 使用上面的处理器进行MNIST数据的处理，并加载
    3. 将每一张图片的标签转换成one-hot向量
'''
dataPath = "F:\\Code_Set\\Python\\PaperExp\\DataSetForPaper\\" #在使用的时候请改成自己实际的MNIST数据集路径
mnistTrain = datasets.MNIST(dataPath, train = True,  download = False, transform = picProcessor)
mnistTest = datasets.MNIST(dataPath, train = False, download = False, transform = picProcessor)

# 因为如果在CPU上，模型的训练速度还是相对来说较慢的，所以如果有条件的话就在GPU上跑吧（一般的N卡基本都支持）
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

'''
神经网络类的定义
    1. C1(卷积层): in_channel = 1, out_channel = 6, kernel_size = (5, 5), stride = 1, 我们在这里将图片进行padding放大：
        padding = 2, padding_mode = 'replicate', 含义是用复制的方式进行padding
    2. 激活函数: 无
    
    3. S2(下采样，即池化层)：kernel_size = (2, 2), stride = (2, 2), in_channel = 6, 采用平均池化，根据论文，加权平均权重及偏置也可训练
    4. 激活函数：1.7159Tanh(2/3 * x)
    
    5. C3(卷积层): in_channel = 6, out_channel = 16, kernel_size = (5, 5), stride = 1, padding = 0, 需要注意的是，这个卷积层
        需要使用map进行一个层次的选择
    6. 激活函数: 无
    
    7. S4(下采样，即池化层)：和S2基本一致，in_channel = 16
    8. 激活函数: 同S2
    
    9. C5(卷积层): in_channel = 16, out_channel = 120, kernel_size = (5, 5), stride = 1, padding = 0
    10. 激活函数: 无
    
    11. F6(全连接层): 120 * 84
    12. 激活函数: 同S4
    
    13. output: RBF函数，定义比较复杂，直接看程序
    无激活函数
    
    按照论文的说明，需要对网络的权重进行一个[-2.4/F_in, 2.4/F_in]的均匀分布的初始化
    
    由于池化层和C3卷积层和Pytorch提供的API不一样，并且RBF函数以及损失函数Pytorch中并未提供，所以我们需要继承nn.Module类自行构造
'''

# 池化层的构造
class Subsampling(nn.Module):
    def __init__(self, in_channel):
        super(Subsampling, self).__init__()

        self.pool = nn.AvgPool2d(2) #先做一个平均池化，然后直接对池化结果做一个加权
        #这个从数学公式上讲和对池化层每一个单元都定义一个相同权重值是等价的

        self.in_channel = in_channel
        F_in = 4 * self.in_channel
        self.weight = nn.Parameter(torch.rand(self.in_channel) * 4.8 / F_in - 2.4 / F_in, requires_grad = True)
        self.bias = nn.Parameter(torch.rand(self.in_channel), requires_grad = True)

    def forward(self, x):
        x = self.pool(x)
        outs = [] #对每一个channel的特征图进行池化，结果存储在这里

        for channel in range(self.in_channel):
            out = x[:, channel] * self.weight[channel] + self.bias[channel] #这一步计算每一个channel的池化结果[batch_size, height, weight]
            outs.append(out.unsqueeze(1)) #把channel的维度加进去[batch_size, channel, height, weight]
        return torch.cat(outs, dim = 1)


# C3卷积层的构造
class MapConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 5):
        super(MapConv, self).__init__()

        #定义特征图的映射方式
        mapInfo = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        mapInfo = torch.tensor(mapInfo, dtype = torch.long)
        self.register_buffer("mapInfo", mapInfo) #在Module中的buffer中的参数是不会被求梯度的

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.convs = {} #将每一个定义的卷积层都放进这个字典

        #对每一个新建立的卷积层都进行注册，使其真正成为模块并且方便调用
        for i in range(self.out_channel):
            conv = nn.Conv2d(mapInfo[:, i].sum().item(), 1, kernel_size)
            convName = "conv{}".format(i)
            self.convs[convName] = conv
            self.add_module(convName, conv)

    def forward(self, x):
        outs = [] #对每一个卷积层通过映射来计算卷积，结果存储在这里

        for i in range(self.out_channel):
            mapIdx = self.mapInfo[:, i].nonzero().squeeze()
            convInput = x.index_select(1, mapIdx)
            convOutput = self.convs['conv{}'.format(i)](convInput)
            outs.append(convOutput)
        return torch.cat(outs, dim = 1)


# RBF函数output层的构建
class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, init_weight = None):
        super(RBFLayer, self).__init__()
        if init_weight is not None:
            self.register_buffer("weight", torch.tensor(init_weight))
        else:
            self.register_buffer("weight", torch.rand(in_features, out_features))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = (x - self.weight).pow(2).sum(-2)
        return x


# 损失函数的构建
def loss_fn(pred, label):
    if(label.dim() == 1):
        return pred[torch.arange(pred.size(0)), label]
    else:
        return pred[torch.arange(pred.size(0)), label.squeeze()]


# RBF的初始化权重
_zero = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_one = [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_two = [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, -1, -1, -1, -1, +1, +1] + \
       [-1, -1, -1, -1, +1, +1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_three = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_four = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1]

_five = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [+1, +1, +1, +1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_six = [-1, -1, +1, +1, +1, +1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_seven = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_eight = [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_nine = [-1, +1, +1, +1, +1, +1, -1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, +1, +1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]


RBF_WEIGHT = np.array([_zero, _one, _two, _three, _four, _five, _six, _seven, _eight, _nine]).transpose()

#整个神经网络的搭建
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1, 6, 5, padding = 2, padding_mode = 'replicate')
        self.S2 = Subsampling(6)
        self.C3 = MapConv(6, 16, 5)
        self.S4 = Subsampling(16)
        self.C5 = nn.Conv2d(16, 120, 5)
        self.F6 = nn.Linear(120, 84)
        self.Output = RBFLayer(84, 10, RBF_WEIGHT)

        self.act = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                F_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in
            elif isinstance(m, nn.Linear):
                F_in = m.in_features
                m.weight.data = torch.rand(m.weight.data.size()) * 4.8 / F_in - 2.4 / F_in

    def forward(self, x):
        x = self.C1(x)
        x = 1.7159 * self.act(2 * self.S2(x) / 3)
        x = self.C3(x)
        x = 1.7159 * self.act(2 * self.S4(x) / 3)
        x = self.C5(x)

        x = x.view(-1, 120)

        x = 1.7159 * self.act(2 * self.F6(x) / 3)

        out = self.Output(x)
        return out

lossList = []
trainError = []
testError = []

#训练函数部分
def train(epochs, model, optimizer, scheduler: bool, loss_fn, trainSet, testSet):

    trainNum = len(trainSet)
    testNum = len(testSet)
    for epoch in range(epochs):
        lossSum = 0.0
        print("epoch: {:02d} / {:d}".format(epoch+1, epochs))

        for idx, (img, label) in enumerate(trainSet):
            x = img.unsqueeze(0).to(device)
            y = torch.tensor([label], dtype = torch.long).to(device)

            out = model(x)
            optimizer.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            lossSum += loss.item()
            if (idx + 1) % 2000 == 0: print("sample: {:05d} / {:d} --> loss: {:.4f}".format(idx+1, trainNum, loss.item()))

        lossList.append(lossSum / trainNum)

        with torch.no_grad():
            errorNum = 0
            for img, label in trainSet:
                x = img.unsqueeze(0).to(device)
                out = model(x)
                _, pred_y = out.min(dim = 1)
                if(pred_y != label): errorNum += 1
            trainError.append(errorNum / trainNum)

            errorNum = 0
            for img, label in testSet:
                x = img.unsqueeze(0).to(device)
                out = model(x)
                _, pred_y = out.min(dim = 1)
                if(pred_y != label): errorNum += 1
            testError.append(errorNum / testNum)

        if scheduler == True:
            if epoch < 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1.0e-3
            elif epoch < 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5.0e-4
            elif epoch < 15:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2.0e-4
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1.0e-4

    torch.save(model.state_dict(), 'F:\\Code_Set\\Python\\PaperExp\\LeNet-5\\epoch-{:d}_loss-{:.6f}_error-{:.2%}.pth'.format(epochs, lossList[-1], testError[-1]))


if __name__ == '__main__':

    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 1.0e-3)

    scheduler = True

    epochs = 25

    train(epochs, model, optimizer, scheduler, loss_fn, mnistTrain, mnistTest)
    plt.subplot(1, 3, 1)
    plt.plot(lossList)
    plt.subplot(1, 3, 2)
    plt.plot(trainError)
    plt.subplot(1, 3 ,3)
    plt.plot(testError)
    plt.show()
