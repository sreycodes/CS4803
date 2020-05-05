import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from core.q_train import QNTrain


class DQNTrain(QNTrain):
    """
    Class for training a DQN
    """
    def __init__(
        self,
        q_net_class,
        env,
        config,
        device,
        logger=None,
    ):
        super().__init__(env, config, logger)
        self.device = device

        self.q_net = q_net_class(env, config)
        self.target_q_net = q_net_class(env, config)

        self.q_net.to(device)
        self.target_q_net.to(device)

        self.update_target_params()
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        self.target_q_net.eval()

        if config.optim_type == 'adam':
            self.optimizer = optim.Adam(
                self.q_net.parameters(), lr=config.lr_begin)
        elif config.optim_type == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.q_net.parameters(), lr=config.lr_begin)
        else:
            raise ValueError(f"Unknown optim_type: {config.optim_type}")


    def process_state(self, state):
        """
        Processing of state

        Args:
            state: np.ndarray of shape either (batch_size, H, W, C)
            or (H, W, C), of dtype 'np.uint8'

        Returns:
            state: A torch float tensor on self.device of shape
            (*, H, W, C), where * = batch_size if it was present in
            input, 1 otherwise. State is normalized by dividing by
            self.config.high
        """
        
        norm_state = state / self.config.high
        if len(norm_state.shape) != 4:
            H, W, C = norm_state.shape
            norm_state = np.reshape(norm_state, (1, H, W, C))
        return torch.from_numpy(norm_state).float()


    def forward_loss(
        self,
        state,
        action,
        reward,
        next_state,
        done_mask,
    ):
        """
        Compute loss for a batch of transitions. Transitions are defiend as
        tuples of (state, action, reward, next_state, done).

        Args:
            state: batch of states (batch_size, *)
            action: batch of actions (batch_size)
            next_state: batch of next states (batch_size, *)
            reward: batch of rewards (batch_size)
            done_mask: batch of boolean values, 1 if next state is terminal
                state (ending the episode) and 0 otherwise.

        Returns:
            The loss for a transition is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2

            Notation:
                s, s': current and next state respectively
                a: current action
                a': possible future actions at s'
                Q: self.q_net
                Q_target: self.target_q_net
        """

        q_net_result = self.q_net(self.process_state(state))
        target_q_net_result = self.target_q_net(self.process_state(next_state))

        batch_size, num_actions = q_net_result.shape
        gamma = 0.9

        return nn.functional.mse_loss(input=torch.from_numpy(reward) + gamma * target_q_net_result.max(1)[0].unsqueeze(1) * done_mask, \
            target=q_net_result.gather(1, torch.from_numpy(action).long().view(-1, 1)))


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.target_q_net.load_state_dict(self.q_net.state_dict())


    def module_grad_norm(self, net):
        """
        Compute the L2 norm of gradients accumulated in net
        """
        with torch.no_grad():
            total_norm = 0
            for param in net.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = np.sqrt(total_norm)
            return total_norm


    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        torch.save(self.q_net.state_dict(),
            os.path.join(self.config.model_output,
                f"{self.q_net.__class__.__name__}.vd"))


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        state = self.process_state(state)
        action_values = self.q_net(state)

        return np.argmax(action_values.cpu().numpy()), action_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate

        Returns:
            q_loss: Loss computed using self.forward_loss
            grad_norm_eval: L2 norm of self.q_net gradients, computed
                using self.module_grad_norm
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch \
            = replay_buffer.sample(self.config.batch_size)

        q_loss = self.forward_loss(s_batch, a_batch, r_batch, sp_batch, done_mask_batch)
        self.q_net.train()
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        return q_loss, self.module_grad_norm(self.q_net)
