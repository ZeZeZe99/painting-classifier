import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        # print(x.shape) # [25, 3, 256, 256]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.shape) # [25, 10, 127, 127]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape) # [25, 20, 62, 62]
        x = x.view(x.shape[0],-1)
        # print(x.shape) # [25, 76880]
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x