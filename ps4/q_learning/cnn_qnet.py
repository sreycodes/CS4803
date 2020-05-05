import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        H, W, C = env.observation_space.shape
        csh = config.state_history
        na = env.action_space.n

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=C * csh, out_channels=32, kernel_size=(H // 20, W // 20), stride=H //  20),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=7*7*64, out_features=256),
            nn.ReLU())
        self.fc2 = nn.Linear(in_features=256, out_features=na)

    def forward(self, state):
        

        out1 = self.conv1(state.permute(0, 3, 1, 2))
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.fc1(out3.view(-1, 7*7*64))
        out = self.fc2(out4)
        
        return out
