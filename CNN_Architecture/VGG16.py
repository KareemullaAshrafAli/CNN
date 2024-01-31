import torch 
import torch.nn as nn
import torch.nn.functional as F






"""
Input Sizes: 224 x 224 x 3

Conv1:  224 x 224 x 64
Conv2:  224 x 224 x 64

MaxPool1: 112 x 112 x 64

Conv3:  112 x 112 x 128
Conv4:  112 x 112 x 128

MaxPool2: 56 x 56 x 128

Conv5:  56 x 56 x 256
Conv6:  56 x 56 x 256
Conv7:  56 x 56 x 256

MaxPool3: 28 x 28 x 256

Conv8:  28 x 28 x 512
Conv9:  28 x 28 x 512
Conv10:  28 x 28 x 512

MaxPool4: 14 x 14 x 512
Conv11:  14 x 14 x 512
Conv12:  14 x 14 x 512
Conv13:  14 x 14 x 512

MaxPool5: 7 x 7 x 512


"""




class VGG16(nn.Module):
    def __init__(self, num_classes:int, dropout:float):
        super(VGG16, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv9 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.maxpool4 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv12 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv13 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.maxpool5 = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        
        self.num_classes = num_classes
        self.dropout = dropout        
        
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)
        
        self.b1 = nn.BatchNorm2d(64)
        self.b2 = nn.BatchNorm2d(64)
        self.b3 = nn.BatchNorm2d(128)
        self.b4 = nn.BatchNorm2d(128)
        self.b5 = nn.BatchNorm2d(256)
        self.b6 = nn.BatchNorm2d(256)
        self.b7 = nn.BatchNorm2d(256)
        self.b8 = nn.BatchNorm2d(512)
        self.b9 = nn.BatchNorm2d(512)
        self.b10 = nn.BatchNorm2d(512)
        self.b11 = nn.BatchNorm2d(512)
        self.b12 = nn.BatchNorm2d(512)
        self.b13 = nn.BatchNorm2d(512)
        
#         MaxPool(ReLU(BatchNorm(Convolution)))
        
    def forward(self, x):
        
        x = F.relu(self.b1(self.conv1(x)))
        x = self.maxpool1(F.relu(self.b2(self.conv2(x))))
        
        x = F.relu(self.b3(self.conv3(x)))
        x = F.relu(self.b4(self.conv4(x)))
        x = self.maxpool2(x)
        
        x = F.relu(self.b5(self.conv5(x)))
        x = F.relu(self.b6(self.conv6(x)))
        x = F.relu(self.b7(self.conv7(x)))
        x = self.maxpool3(x)
        
        x = F.relu(self.b8(self.conv8(x)))
        x = F.relu(self.b9(self.conv9(x)))
        x = F.relu(self.b10(self.conv10(x)))
        x = self.maxpool4(x)
        
        x = F.relu(self.b11(self.conv11(x)))
        x = F.relu(self.b12(self.conv12(x)))
        x = F.relu(self.b13(self.conv13(x)))
        x = self.maxpool5(x)
        
        y = nn.Flatten()
        x = y(x)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = self.dropout)        
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p = self.dropout)
        
        x = nn.Softmax(self.fc3(x))
        
        return x