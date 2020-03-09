import torch

class CNN(torch.nn.Module):
    def __init__(self,num_class):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),

        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

        )
        self.fc = torch.nn.Linear(1600,10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # flatten
        x = x.view(x.size(0),-1)
        fc_rep = x
        return fc_rep,self.fc(x)



