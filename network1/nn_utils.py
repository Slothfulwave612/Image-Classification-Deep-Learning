'''
nn_utils.py
-----------

This module includes some function which will be used by the module 'network1.py'.

Functions included(4):
===================
1. sigmoid: implements the sigmoid operation.
2. relu: implements the relu operation.
3. sigmoid_backward: implements the backpropagation for single sigmoid unit.
4. relu_backward: implements the backpropagation for single relu unit.

Modules used(1):
=============
1. numpy: package for scientific computing with Python.
'''

import numpy as np

def sigmoid(Z):
    '''
    Implements the sigmoid activation function.

    Arguments:
    Z -- numpy array 

    Returns:
    A -- output of sigmoid(Z)
    cache -- returns Z as well, used during backpropagation
    '''

    A = 1.0 / (1.0 + np.exp(-Z))

    assert(A.shape == Z.shape)
    ## assertion for checking the shape of A matches Z or not

    cache = Z
    return A, cache

def relu(Z):
    '''
    Implements the ReLu activation function.

    Arguments:
    Z -- numpy array 

    Returns:
    A -- output of max(0,Z)
    cache -- returns Z as well, used during backpropagation
    '''

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)
    ## assertion for checking the shape of A matches Z or not

    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    pass

def relu_backward(dA, cache):
    pass

