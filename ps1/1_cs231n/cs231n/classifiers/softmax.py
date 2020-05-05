import numpy as np
from random import shuffle
from scipy.special import softmax

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.

  loss = 0.0
  dW = reg * W

  N = X.shape[1]

  pred = softmax(np.dot(W, X), axis=0)
  L = np.mean(-np.log(pred[y, np.arange(N)] + 1e-8))
  R = 0.5 * reg * np.sum(W ** 2)
  loss = L + R

  d1 = np.copy(pred)
  d1[y, np.arange(N)] -= 1
  d1 /= N
  dW += np.dot(d1, X.T)
  
  return loss, dW
