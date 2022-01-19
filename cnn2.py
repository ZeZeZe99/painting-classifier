"""
This class is a CNN model
"""
from torch import nn


class CNN2(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            # original: 256x256x3

            # output: 256x256x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 256x256x16
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 128x128x16
            nn.MaxPool2d(kernel_size=2),
            # output: 128x128x32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 128x128x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 128x128x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 64x64x32
            nn.MaxPool2d(kernel_size=2),
            # output: 64x64x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 64x64x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 64x64x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 32x32x64
            nn.MaxPool2d(kernel_size=2),
            # output: 32x32x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 32x32x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 32x32x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 16x16x256
            nn.MaxPool2d(kernel_size=2),
            # output: 16x16x256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 16x16x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 16x16x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 8x8x256
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(8*8*256, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1584),
            nn.Softmax(1),
            nn.Linear(1584, output_dim)


        )

    def forward(self, x):
        return self.network(x)