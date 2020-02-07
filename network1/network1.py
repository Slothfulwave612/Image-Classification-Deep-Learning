'''
network1.py
-----------

This module which contains all the helper functions for our neural network.
The module includes seven main functions which will be used by our neural
network to perform various operation like forward propagation, backward
propagation and updation of parameters.

We are making a 2 layer neural n/w, i.e. a neural network having one input
layer, 1 hidden layer and 1 output layer.

Functions included(8):
===================
1. initialize_parameters: will initialize our W1, b1, W2 and b2.
2. linear_forward: will compute Z(= W.X + b).
3. linear_activation_forward: will compute a(z), either ReLU or sigmoid.
4. compute_cost: will compute the cost(J), cross-entropy cost function.
5. linear_backward: will compute dW, db, dA_prev.
6. linear_activation_backward: backpropagation process for our neural network.
7. update_parameters: for updating out parameters(W and b).
8. save_parameters: for saving weights and biases values.

Modules used(2):
=============
1. numpy: package for scientific computing with Python.
2. nn_utils: provides some necessary functions for this module.
'''

import numpy as np
from nn_utils import sigmoid, sigmoid_backward, relu, relu_backward

def initialize_parameters(n_x, n_h, n_y):
    '''
    Used for initializing the values for our parameters(W and b)

    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- a dictionary containing the parameters:
                  W1 -- weight matrix of shape(n_h, n_x) 
                  b1 -- bias matrix of shape(n_h, 1)
                  W2 -- weight matrix of shape(n_y, n_h)
                  b2 -- bias matrix of shape(n_y, 1)
    '''

    np.random.seed(1)
    ## makes the random numbers predictable
    ## with the seed value, the same set of numbers will appear every time.

    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))
    ## using random.randn and zeros method 
    ## initializing the parameters values

    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    ## assertion for checking the shape of matrices

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2, 
                  'b2': b2}
    
    return parameters

def linear_forward(A, W, b):
    '''
    Implements the linear part of a layer's forward propagation

    Arguments:
    A -- activation from previous layer(or i/p data)
    W -- weight matrix
    b -- bias matrix

    Returns:
    Z -- the input activation function(W.A + b)
    cache -- tuple containing 'A', 'W' and 'b', 
             stored for computing backward pass efficiently
    '''

    Z = np.dot(W,A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    ## assertion for checking the shape of Z

    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
    Implements the forward propagation for the Linear->Activation layer.

    Arguments:
    A_prev -- activation from previous layer(or input data)
    W -- weight matrix
    b -- bias matrix
    activation -- activation used in this layer, 'relu' or 'sigmoid

    Returns:
    A -- the output of activation function
    cache -- tuple containing 'linear_cache' and 'activation_cache',
             stored for computing backward pass efficiently
    '''

    Z, linear_cache = linear_forward(A_prev, W, b)
    ## calling linear_forward function
    ## to get the value of Z and linear cache

    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
        ## calling sigmoid function defined in nn_utils
    
    elif activation == 'relu':
        A, activation_cache = relu(Z)
        ## calling relu function defined in nn_utlis

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    ## assertion for checking the shape of A

    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL, Y):
    '''
    Implements the cross-entropy cost function

    Arguments:
    AL -- predicted output from the last layer
    Y -- true label

    Returns:
    cost -- cross-entropy cost
    '''

    m = Y.shape[1]

    cost = (-1.0/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    ## cross-entropy cost function

    cost = np.squeeze(cost)    
    ## this method turns [[10]] into [10]

    assert(cost.shape == ())
    ## assertion condition for checking the shape of cost

    return cost
