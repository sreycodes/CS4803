import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
        self.c, self.h, self.w = im_size
        self.w1 = nn.Linear(self.c * self.h * self.w, hidden_dim)
        self.a1 = nn.ReLU(inplace=True)
        self.w2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = self.w2(self.a1(self.w1(images.view(-1, self.c * self.h * self.w))))
        return scores

