#https://blog.csdn.net/song5bai/article/details/116358451
#1.引入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#2.读入数据
#一些参数
batch_size_train = 200
batch_size_test = 1000
learning_rate = 0.0073564225
momentum = 0.9
random_seed = 1
torch.manual_seed(random_seed)
 
transform = transforms.Compose(
    [transforms.ToTensor(),# PIL Image → Tensor
    transforms.Normalize((0.1307,),(0.3081,))]  # 0,1 →  -1,1
)
 
trainset = torchvision.datasets.MNIST(download=True,root='./data',train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train,shuffle=True,num_workers=2)
testset = torchvision.datasets.MNIST(download=True,root='./data',train=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size_test,shuffle=True,num_workers=2)
 
# explore testing data
examples = enumerate(trainloader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

#展示数据（示例）：
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(8):
  plt.subplot(2,4,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
#3.定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*5*5,120) #3-dim (0,1,2)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))#
        #print(x.size()) # 4*16*5*5
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

#GPU（示例）：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)

#优化设置（示例）：
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum)
# 损失函数
criterion = nn.CrossEntropyLoss()

#4.训练
epochs=15
for epoch in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images).to(device)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%200 == 199:
            print(f'{epoch+1}, {i+1}; loss:{running_loss/200}')
            running_loss = 0.0
print('Finished Traing')

#5.在测试集上测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
print(f'Accuracy :{100*correct/total} %')

#各类的精确度（示例）：
classes = (0,1,2,3,4,5,6,7,8,9)
class_correct = [0 for _ in range(10)]
class_total = [0 for _ in range(10)]
with torch.no_grad():
    for data in testloader:
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        c = (predicted==labels).squeeze() # 1*4 → 4
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %3s : %2d %%'%(classes[i],100*class_correct[i]/class_total[i]))

#保存和加载（示例）：
PATH = "./data/MNIST.model"
#快速保存我们训练过的模型
torch.save(net.state_dict(), PATH)
 
#重新加载保存的模型
net.load_state_dict(torch.load(PATH))
