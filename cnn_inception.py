import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                  nn.ReLU())

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU())

    def forward(self, input):
        output1 = self.branch1(input)
        output2 = self.branch2(input)
        output3 = self.branch3(input)
        output4 = self.branch4(input)
        return torch.cat([output1, output2, output3, output4], dim=1)

if __name__ == '__main__':
    model = InceptionModule(in_channels=3, out_channels=32)
    inp = torch.rand(1, 3, 128, 128)
    print(model(inp).shape)