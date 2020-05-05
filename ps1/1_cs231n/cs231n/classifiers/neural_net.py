import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  fc1 = np.dot(X, W1) + b1
  relu = np.maximum(0, fc1)
  scores = np.dot(relu, W2) + b2
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  C = np.max(y) + 1
  _y = np.zeros((N, C))
  _y[np.arange(N), y] = 1

  # compute the loss
  loss = None
  pred = softmax(scores, axis=1)
  # print(pred)
  # print(pred[np.arange(N), y])
  # print(np.log10(pred[np.arange(N), y]))
  L = np.mean(-np.log(pred[np.arange(N), y]))
  # print(L)
  R1 = np.sum(W1 ** 2)
  R2 = np.sum(W2 ** 2)
  R = 0.5 * reg * (R1 + R2)
  # print(R1)
  # print(R2)
  # print(R)
  loss = L + 0.5 * reg * (R1 + R2)

  # compute the gradients
  grads = {}
  d1 = pred
  d1[np.arange(N), y] -= 1
  d1 /= N
  grads['W2'] = np.dot(relu.T, d1) + reg * W2
  grads['b2'] = np.sum(d1.T, axis=1)
  dH = np.dot(d1, W2.T)
  dH[relu <= 0] = 0
  grads['W1'] = np.dot(X.T, dH) + reg * W1
  grads['b1'] = np.sum(dH.T, axis=1)


  return loss, grads

