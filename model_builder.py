import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(810, 50),
            nn.ReLU(),
            nn.Linear(50, 7)
        )
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(),
            nn.Linear(32, 3*2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        out = self.stn(input)
        out = self.block1(out)
        out = self.block2(out)
        out = F.dropout(out)
        out = out.view(-1, 810)
        out = self.block3(out)

        return out


