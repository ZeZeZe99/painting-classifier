from torch import nn


# Simplified CNN2
class CNN10(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            # original: 256x256x3

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2),
            # nn.ReLU(),

            nn.Flatten(),
            nn.Linear(512 * 256, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.network(x)
