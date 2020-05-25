import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse

# 定義是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義網路結構
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保證輸入輸出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定義前向傳播過程，輸入為x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的 I/O都是維度為一的值，所以要把多維度的 tensor展平成一維
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #模型保存路徑
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  #模型加載路徑
opt = parser.parse_args()

# 超参數設置
EPOCH = 8   #遍歷數據集次數
BATCH_SIZE = 64      #批處理尺寸(batch_size)
LR = 0.001        #學習率

# 定義數據預處理方式
transform = transforms.ToTensor()

# 定義訓練數據集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)

# 定義測試批批處理數據
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定義測試批數據集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# 定義測試批處理數據
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 定义損失函數loss function 和優化方式（採用SGD）
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵損失函數，通常用于多分類問題上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 數據讀取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每訓練100個batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch測試一下準確率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那個類
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的識別準確率為：%d%%' % (epoch + 1, (100 * correct / total)))
    torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))
