import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# from model.DogCatNet import DogCatVGGNet as DogCatNet
class DogCatVGGNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(2)
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)

        self.conv2 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3)

        self.conv4 = torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)

        
        self.conv5 = torch.nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,stride=3)
        self.linear1 = torch.nn.Linear(in_features=25088,out_features=2)
        
    def forward(self,x):
        batch_size = x.size(0)
        #[,64,226,226]
        x = F.relu(self.conv1(x))
        #[,64,224,224]
        x = self.avgpool(x)
        #[,64,112,112]
        x = F.relu(self.conv2(x))
        #[,128,58,58]
        x = F.relu(self.conv3(x))
        #[,128,56,56]
        x = self.avgpool(x)
        #[,128,28,28]
        x = F.relu(self.conv4(x))
        #[,256,24,24]
        x = F.relu(self.conv5(x))
        #[,512,7,7]
        x = x.view(batch_size,-1)
        x = self.linear1(x)
        return x
    
transforms = transforms.Compose([
    transforms.RandomResizedCrop(226),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
batch_size = 64
train_dataset = datasets.ImageFolder("./dataset/catsdogs/train/", transforms)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.ImageFolder("./dataset/catsdogs/test/", transforms)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

model = DogCatVGGNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
 
 
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # 张量之间的比较运算
    return correct/total

if __name__ == '__main__':
    accuracy_max = 0
    accuracy = 0
    accuracys = []
    for epoch in range(1000):
        train(epoch)
        accuracy = test()
        if(accuracy_max < accuracy):
            accuracy_max = accuracy
            print("save: train_accuracy (%d,%d) epoch:%d"%(accuracy* 100,accuracy_max * 100,epoch))
            torch.save(model.state_dict(), "./save/dogcat.pt")
        accuracys.append(accuracy)
    print("accuracy (%d,%d)"%(accuracy* 100,accuracy_max * 100))
    plt.plot(accuracys)