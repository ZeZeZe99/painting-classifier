from torch import nn
from cnn_inception import InceptionModule
from cnn_residual import make_layer
from cnn_residual import BasicBlock

class ZNN2(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(

            InceptionModule(in_channels=3, out_channels=32),
            nn.AvgPool2d(kernel_size=2),

            make_layer(BasicBlock, 128, 128, 1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(4*4*512, 1024),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    x = ZNN2(9)
    print(x)
