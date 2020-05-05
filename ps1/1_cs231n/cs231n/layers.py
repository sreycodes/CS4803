import numpy as np
from itertools import product

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  out = np.dot(np.reshape(x, (x.shape[0], w.shape[0])), w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx = np.reshape(np.dot(dout, w.T), x.shape)
  dw = np.dot(np.reshape(x, (x.shape[0], w.shape[0])).T, dout)
  db = np.sum(dout, axis=0)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0, x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = dout
  dx[x <= 0] = 0
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  stride = conv_param['stride']
  padding = (conv_param['pad'], conv_param['pad'])
  cache = (x, w, b, conv_param)
  x = np.pad(x, ((0, 0), (0, 0), padding, padding))
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  # print(H, W, HH, WW, padding[0])

  H_ = 1 + (H - HH) // stride
  W_ = 1 + (W - WW) // stride
  out = np.zeros((N, F, H_, W_))

  for n, f, r, c in product(range(N), range(F), range(0, 1 + H - HH, stride), range(0, 1 + W - WW, stride)):
    # print(r, c)
    out[n, f, r // stride, c // stride] = np.sum(x[n, :, r:r + HH, c:c + WW] * w[f, :, :, :]) + b[f]
          
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  padding = (conv_param['pad'], conv_param['pad'])
  dx = np.zeros(x.shape)
  x = np.pad(x, ((0, 0), (0, 0), padding, padding))
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, F, H_, W_ = dout.shape
  dx_pad = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  for f, r, c, n in product(range(F), range(H_), range(W_), range(N)):
    dw[f] += x[n, :, r * stride: r * stride + HH, c * stride: c * stride + WW] * dout[n, f, r, c]
    dx_pad[n, :, r * stride : r * stride + HH, c * stride : c * stride + WW] += w[f] * dout[n, f, r, c]
  dx = dx_pad[:, :, padding[0] : -padding[0], padding[0] : -padding[0]]

  db = np.sum(dout, axis=(0, 2, 3))

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  N, C, H, W = x.shape
  pH, pW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  out = np.zeros((N, C, H // stride, W // stride))
  for n, c, r, col in product(range(N), range(C), range(0, H - pH + 1, stride), range(0, W - pW + 1, stride)):
    out[n, c, r // stride, col // stride] = np.max(x[n, c, r : r + pH, col : col + pW])
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  dx = np.zeros(x.shape)
  N, C, H, W = x.shape
  pH, pW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  for n, c, r, col in product(range(N), range(C), range(0, H - pH + 1, stride), range(0, W - pW + 1, stride)):
    idx = np.argmax(x[n, c, r : r + pH, col : col + pW])
    dx[n, c, r + idx // pW, col + idx % pW] = dout[n, c, r // stride, col // stride]
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

