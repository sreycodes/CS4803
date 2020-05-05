import numpy as np

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, env, config):
        """
        A state-action (Q) network with a single fully connected
        layer, takes the state as input and gives action values
        for all actions.
        """
        super().__init__()

        H, W, C = env.observation_space.shape
        csh = config.state_history
        na = env.action_space.n
        self.fc = nn.Linear(H * W * C * config.state_history, na)

    def forward(self, state):
        """
        Returns Q values for all actions

        Args:
            state: tensor of shape (batch, H, W, C x config.state_history)

        Returns:
            q_values: tensor of shape (batch_size, num_actions)
        """
        
        # print(state)
        b, H, W, C = [*state.size()]
        return self.fc(state.reshape(b, H * W * C))
