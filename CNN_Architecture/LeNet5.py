import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = (7,7), stride = (1,1), padding = (1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))

        self.conv2 = nn.Conv2d(6, 16, kernel_size = (7,7), stride = (1,1), padding = (1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))

        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 100) 

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))

        x = F.relu(self.maxpool2(self.conv2(x)))

        return x 