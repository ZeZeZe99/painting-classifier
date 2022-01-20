"""
This class is a CNN model
"""
import torch
from torch import nn
import numpy as np


class CNN3(nn.Module):

    def __init__(self, output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out((3, 256, 256))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_out(self, shape):
        o = self.network(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.float().to(device)

        network_out = self.network(x).view(x.size()[0], -1)
        return self.fc(network_out)
