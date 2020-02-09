'''
nn_utils.py
-----------

This module includes some function which will be used by the module 'network1.py'.

Functions included(5):
===================
1. sigmoid: implements the sigmoid operation.
2. relu: implements the relu operation.
3. sigmoid_backward: implements the backpropagation for single sigmoid unit.
4. relu_backward: implements the backpropagation for single relu unit.
5. load_data: loads the train and test data

Modules used(2):
=============
1. numpy: package for scientific computing with Python.
2. h5py: pythonic interface to the HDF5 binary data format.
'''

import numpy as np
import h5py

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
    '''
    Implements backward propagation for single sigmoid unit

    Arguments:
    dA -- post-activation gradient
    cache -- the Z values

    Returns:
    dZ -- gradient of cost w.r.t Z
    '''

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)

    assert(dZ.shape == Z.shape)
    ## assertion condition for checking the shape of dZ

    return dZ

def relu_backward(dA, cache):
    '''
    Implements backward propagation for single relu unit

    Arguments:
    dA -- post-activation gradient
    cache -- the Z values

    Returns:
    dZ -- gradient of cost w.r.t Z
    '''

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0 

    assert(dZ.shape == Z.shape)
    ## assertion condition for checking the shape of dZ

    return dZ

def load_data():
    '''
    For loading the dataset, train as well as test

    Returns:
    train_set_x_orig -- the training dataset with feature value
    train_set_y_orig -- the traning dataset with label value
    test_set_x_orig -- the test dataset with feature value
    test_set_y_orig -- the test dataset with label value
    '''

    train_dataset = h5py.File(r'../data/train_catvnoncat.h5', 'r')
    ## reading from h5 file(train_catvnoncat.h5)
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    ## train set featuers
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    ## train set labels

    test_dataset = h5py.File('../data/test_catvnoncat.h5', 'r')
    ## reading from h5 file(test_catvnoncat.h5)
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    ## test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])
    ## test set labels

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    ## the above code will convert shape (14,) to (1,14)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

## slothfulwave612