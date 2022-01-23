"""
This class is a CNN model
"""
from torch import nn


class CNN1(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            # original: 256x256x3

            # output: 256x256x64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 256x256x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 128x128x64
            nn.MaxPool2d(kernel_size=2),
            # output: 128x128x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 128x128x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 64x64x128
            nn.MaxPool2d(kernel_size=2),
            # output: 64x64x256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 64x64x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 64x64x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 32x32x256
            nn.MaxPool2d(kernel_size=2),
            # output: 32x32x512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 32x32x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 32x32x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 16x16x512
            nn.MaxPool2d(kernel_size=2),
            # output: 16x16x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 16x16x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 16x16x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # output: 8x8x512
            nn.MaxPool2d(kernel_size=2),
            # output: 8x8x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # output: 4x4x512
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(4*4*512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.network(x)