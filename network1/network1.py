'''
network1.py
-----------

This module which contains all the helper functions for our neural network.
The module includes seven main functions which will be used by our neural
network to perform various operation like forward propagation, backward
propagation and updation of parameters.

We are making a 2 layer neural n/w, i.e. a neural network having one input
layer, 1 hidden layer and 1 output layer.

Functions included(9):
===================
1. initialize_parameters: will initialize our W1, b1, W2 and b2.
2. linear_forward: will compute Z(= W.X + b).
3. linear_activation_forward: will compute a(z), either ReLU or sigmoid.
4. compute_cost: will compute the cost(J), cross-entropy cost function.
5. linear_backward: will compute dW, db, dA_prev.
6. linear_activation_backward: backpropagation process for our neural network.
7. update_parameters: for updating out parameters(W and b).
8. forward_prop: implements forward propagation.
9. predict: predict the result.

Modules used(2):
================
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
    b -- bias vector

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

    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    ## cross-entropy cost function

    cost = np.squeeze(cost)    
    ## this method turns [[10]] into [10]

    assert(cost.shape == ())
    ## assertion condition for checking the shape of cost

    return cost

def linear_backward(dZ, cache):
    '''
    Implements the linear portion of backward propagation for a single layer

    Arguments:
    dZ -- gradient of cost function w.r.t. the linear output
    cache -- tuple values (A_prev, W, b) coming from forward propagation

    Returns:
    dA_prev -- gradient of cost function w.r.t. the activation of previous layer
    dW -- gradient of cost function w.r.t. W
    db -- gradient of cost function w.r.t. b
    '''

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ,A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    ## calculating dW, db and dA_prev

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    ## assertion conditions

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    '''
    Implement the backward propagation for the Linear->Activation layer

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple values (linear_cache, activation_cache)
    activation -- the activation to be used in this layer, 'relu' or 'sigmoid'

    Returns:
    dA_prev -- gradient of cost w.r.t. the activation
    dW -- gradient of cost w.r.t W
    db -- gradient of cost w.r.t. b
    '''
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        ## relu_backward defined in nn_utils
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        ## sigmoid_backward defined in nn_utils
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    '''
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradient

    Returns:
    parameters -- python dictionary containing updated parameters
    '''
    
    L = len(parameters) // 2
    ## number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def forward_prop(X, parameters):
    '''
    Implements forward propagation

    Arguments:
    X -- data
    parameters -- value of your parameters

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
              the cache of linear_relu_forward()
              the cache of linear_sigmoid_forward()
    '''

    caches = []
    A = X

    L  = len(parameters) // 2
    ## number of layers in the neural network

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))
    ## assertion condition

    return AL, cache

def predict(X, y, parameters, accuracy):
    '''
    This function is used to predict the result 

    Arguments:
    X -- data set of examples 
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    '''

    m = X.shape[1]
    p = np.zeros((1,m))

    probas, caches = forward_prop(X, parameters)
    ## forward propagation

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print(f'{accuracy}: {np.sum((p == y)/m)}')

## slothfulwave612