'''
nn_utils.py
-----------
This module includes some function which will be used by the module 'network2.py'.

Functions included(4):
===================
1. sigmoid: implements the sigmoid operation.
2. relu: implements the relu operation.
3. sigmoid_backward: implements the backpropagation for single sigmoid unit.
4. relu_backward: implements the backpropagation for single relu unit.

Modules used(2):
=============
1. numpy: package for scientific computing with Python.
2. h5py: pythonic interface to the HDF5 binary data format.
'''

import numpy as np
import h5py

def relu(Z):
    '''
    Implements relu-activation function.

    Arguments:
    Z -- numpy array

    Returns:
    A -- output of max(0,Z)
    cache -- contains the value of Z
    '''
    
    A = np.maximum(0, Z)

    cache = Z
    return A, cache

def sigmoid(Z):
    '''
    Implements sigmoid-activation function.

    Arguments:
    Z -- numpy array

    Returns:
    A -- output of sigmoid function
    cache -- value of Z
    '''

    A = 1 / (1 + np.exp(-Z))

    cache = Z
    return A, cache

def relu_backward(dA, cache):
    '''
    Implements the backward propagation for a single ReLU.

    Arguments:
    dA -- post-activation gradient
    cache -- Z value

    Returns:
    dZ -- gradient of cost with respect to Z
    '''

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):
    '''
    Implements the backward propagatio for a single sigmoid unit.

    Arguments:
    dA -- post-activation gradient
    cache -- Z values

    Returns:
    dZ -- gradient of cost function with respect to Z
    '''

    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ

## slothfulwave612