import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        self.c, self.h, self.w = im_size
        stride = 1
        padding = (kernel_size - 1) // 2
        pool_size = 2
        self.conv = nn.Conv2d(self.c, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.affine = nn.Linear(hidden_dim * (self.h // pool_size) * (self.w // pool_size), n_classes)

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        scores = self.affine(self.pool(self.act(self.conv(images))).view(images.shape[0], -1))
        return scores

