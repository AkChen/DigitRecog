import torch
import torch.nn as nn


class FusionNet(nn.Module):

    def __init__(self, mnist_net, svhn_net,num_classes):
        super(FusionNet, self).__init__()

        self.mnist_net = mnist_net
        self.svhn_net = svhn_net
        self.num_classes = num_classes

        self.fc1 = nn.Linear(1024+1600,512)
        self.fc2 = nn.Linear(512,10)


    def forward(self, mnist,svhn):

        mnist_rep = self.mnist_net(mnist)[0]
        svhn_rep = self.svhn_net(svhn)[0]

        concat_rep = torch.cat((mnist_rep,svhn_rep),dim = 1)

        concat_rep = self.fc1(concat_rep)
        concat_rep = torch.relu(concat_rep)
        concat_rep = self.fc2(concat_rep)
        concat_rep = torch.relu(concat_rep)

        return concat_rep



