import torch
import torch.nn as nn
import torch.nn.functional as F 



class AlexNet(nn.Module):
    def __init__(self, num_classes:int, dropout:float):
        super(AlexNet, self).__init__()
        
        self.num_classes = num_classes
        self.dr = dropout
        
        self.conv1 = nn.Conv2d(3, 96, (7,7), (4,4), (0,0) )
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2) )
        
        self.conv2 = nn.Conv2d(96, 256, (3,3), (1,1), (1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2))
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
        self.conv4 = nn.Conv2d(384, 384, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
        self.conv5 = nn.Conv2d(384, 256, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2))
        
        self.linear1 = nn.Linear(9216, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, self.num_classes)
        self.softmax = nn.LogSoftmax(dim = 1)
        
        
        
    def forward(self, x):
        
        x = self.maxpool1(F.relu(self.conv1(x)))
    
        x = self.maxpool2(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = self.maxpool3(F.relu(self.conv5(x)))
        
        x = x.reshape(x.shape[0], -1)
        
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p = self.dr)
        
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p = self.dr)
        
        x = self.softmax(self.linear3(x))        
        
        return x
    
"""
nn.Linear()
nn.Conv2d()
nn.LogSoftmax()
nn.Softmax()
F.relu()
F.dropout()
"""   