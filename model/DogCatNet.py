import torch
import torch.nn.functional as F

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
        print(x.size(0))
        x = self.linear1(x)
        return x