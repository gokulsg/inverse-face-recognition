import torch
import torch.nn.functional as F
from torch import nn

class Generator(nn.Module):
    def __init__(self, n_hidden, bottom_width=4, channels=512):
        super().__init__()
        self.channels = channels
        self.bottom_width = bottom_width

        self.linear = nn.Linear(n_hidden, bottom_width*bottom_width*channels)
        self.dconv1 = nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(channels // 2, channels // 4, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(channels // 4, channels // 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(channels // 8, channels // 16, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(channels // 16, 3, 4, 2, 1)

        self.bn0 = nn.BatchNorm1d(bottom_width*bottom_width*channels)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.bn2 = nn.BatchNorm2d(channels // 4)
        self.bn3 = nn.BatchNorm2d(channels // 8)
        self.bn4 = nn.BatchNorm2d(channels // 16)

    def forward(self, x):
        x = F.relu(self.bn0(self.linear(x))).view(-1, self.channels, self.bottom_width, self.bottom_width)
        x = F.relu(self.bn1(self.dconv1(x)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn4(self.dconv4(x)))

        x = torch.sigmoid(self.dconv5(x))

        return x
