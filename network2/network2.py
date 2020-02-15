'''
network2.py
-----------
This module which contains all the helper functions for our neural network.
The module includes nine main functions which will be used by our neural
network to perform various operation like forward propagation, backward
propagation and updation of parameters.
We are making a L layer neural n/w, i.e. a neural network having one input
layer, L-1 hidden layer and 1 output layer.

Functions included(10):
===================
1. initialize_parameters: will initialize our W1, b1, W2 and b2.
2. linear_forward: will compute Z(= W.X + b).
3. linear_activation_forward: will compute a(z), either ReLU or sigmoid.
4. L_model_forward: implement forward propagation.
5. compute_cost: will compute the cost(J), cross-entropy cost function.
6. linear_backward: will compute dW, db, dA_prev.
7. linear_activation_backward: backpropagation process for our neural network.
8. L_layer_backward: implements the backward propagation.
9. update_parameters: for updating out parameters(W and b).
10. predict: predict the result.

Modules used(2):
================
1. numpy: package for scientific computing with Python.
2. nn_utils: provides some necessary functions for this module.
'''

import numpy as np
import nn_utils

def initialize_paramters(layers_dims):
    '''
    Function to initialize the value of parameters

    Arguments:
    layers_dims -- python array containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters 'W1', 'b1',...., 'WL', 'bL'
    '''

    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters

def linear_forward(A, W, b):
    '''
    Implements the linear part of forward propagation.

    Arguments:
    A -- activations from the previous layer
    W -- weight matrix
    b -- bias matrix

    Returns:
    Z -- the input of the activation function
    cache -- a python dictionary containing 'A', 'W' and 'b', stored for backward pass efficiently
    '''

    Z = np.dot(W, A) + b
    ## calculating Z = W.A + b

    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
    Implements forward propagation.

    Arguments:
    A_prev -- activation from previous layers
    W -- weight matrix
    b -- bias matrix
    activation -- the activation used in this layer, 'sigmoid' or 'relu'

    Returns:
    A -- the output activation 
    cache -- a python dictionary containing 'linear_cache' and 'activation_cache'
    '''
    
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activation_cache = nn_utils.relu(Z)
    
    elif activation == 'sigmoid':
        A, activation_cache = nn_utils.sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    '''
    Implement forward propagation.

    Arguments:
    X -- training features
    parameters -- python dict containing values of our parameters

    Returns:
    AL -- last-post activation value
    caches -- list of cache containing:
              every cache of linear_relu_forward()
              every cache of linear_sigmoid_forward()
    '''

    caches = []
    A = X
    L = len(parameters) // 2            ## number of layers in a neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], 
                                             parameters['b' + str(l)], activation='relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
                                          parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches

def compute_cost(AL, y):
    '''
    Implements the cross-entropy cost function.

    Aruments:
    AL -- the last layer activation(predicted values)
    y -- target labels(true values)

    Returns:
    cost -- cross-entropy cost
    '''

    m = y.shape[1]

    cost = (1/m) * (- np.dot(y, np.log(AL).T) - np.dot(1-y, np.log(1-AL).T))
    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):
    '''
    Implements the linear portion of backward propagation.

    Arguments:
    dZ -- gradeint of cost with respect to the linear output
    cache -- tuple values (A_prev, W, b)

    Returns:
    dA_prev -- gradient of cost with repsect to the activation
    dW -- gradient of cost with respect to W
    db -- gradient of cost with respect to b
    '''

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, current_cache, activation):
    '''
    Implements backward propagation

    Arguments:
    dA -- post activation gradient
    current_cache -- tuple values (linear_cache, activation_cache)
    activation -- activation used in the layer, 'sigmoid' or 'relu'
    '''

    linear_cache, activation_cache = current_cache

    if activation == 'relu':
        dZ = nn_utils.relu_backward(dA, activation_cache) 
    
    elif activation == 'sigmoid':
        dZ = nn_utils.sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_layer_backward(AL, y, caches):
    '''
    Implements the backward propagation

    Arguments:
    AL -- probability vector, output of forward propagation
    y -- true label vector
    caches -- list of caches containing:
              every cache of linear_activation_forward() with 'relu'
              every cache of linear_activation_forward() with 'sigmoid'

    Returns:
    grads -- A dictionary with the gradients
    '''

    grads = {}
    L = len(caches)     ## number of layers
    m = AL.shape[1]
    y = y.reshape(AL.shape)     ## after this line, y will be of same shape as AL

    ## Initializing the backpropagation 
    dAL = - (np.divide(y, AL) - np.divide(1-y, 1-AL))

    current_cache = caches[L-1]
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
                                                               grads['dA' + str(l+1)],
                                                               current_cache,
                                                               'relu'
                                                              )
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l+1)] = dW_temp
        grads['db' + str(l+1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    '''
    Update paramters using gradient descent.

    Arguments:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    '''

    L = len(parameters) // 2    ## number of layers

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - (learning_rate * grads['dW' + str(l+1)])
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - (learning_rate * grads['db' + str(l+1)])

    return parameters

def predict(X, y, parameters):
    '''
    This function is used to predict result of L layer model.

    Aruments:
    X -- the data features
    y -- the true labels
    parameters -- parameters of trained model

    Returns:
    p -- predictions for the given dataset X
    '''

    m = X.shape[1]
    n = len(parameters) // 2    ## number of layers in our neural network
    p = np.zeros((1,m))

    probab, caches = L_model_forward(X, parameters)

    for i in range(probab.shape[1]):
        if probab[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print(f'Accuracy: {np.sum(p == y) / m}')

## slothfulwave612