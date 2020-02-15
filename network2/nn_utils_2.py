'''
nn_utils_2.py
-------------
This module contains some helper function which will be used by run_script.py
to make our Neural Network function.
The function here will explore the dataset, reshape it, train the neural
network and test our model.

Function Used(4):
================
1. load_data: for loading the dataset, train dataset and test dataset.
2. explore_dataset: for exploring the dataset.
3. reshapingX: for reshaping the shape of X for test and train dataset.
4. L_layer_model: main function where neural network is trained.

Modules Used(5):
===============
1. numpy: package for scientific computing with Python.
2. matplotlib.pyplot: plotting library in Python.
3. h5py: pythonic interface to the HDF5 binary data format.
4. network1: provides necessary functions for our neural network. 
5. nn_utils: provides some necessary functions for this module.
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import network2
import nn_utils

def load_data():
    '''
    For loading the dataset, train dataset and test dataset.

    Arguments: zero

    Returns:
    train_X_orig -- the training dataset with feature values
    train_y -- the training dataset with label values
    test_X_orig -- the testing dataset with feature values
    test_y -- the testing dataset with label values
    '''

    train_dataset = h5py.File('../data/train_catvnoncat.h5', 'r')
    ## reading data from h5 file(train_catvnoncat.h5)
    train_X_orig = np.array(train_dataset['train_set_x'][:])
    ## train set features
    train_y_orig = np.array(train_dataset['train_set_y'][:])
    ## train set labels

    test_dataset = h5py.File('../data/test_catvnoncat.h5', 'r')
    ## reading data from h5 file(test_catvnoncat.h5)
    test_X_orig = np.array(test_dataset['test_set_x'][:])
    ## test set features
    test_y_orig = np.array(test_dataset['test_set_y'][:])
    ## test set labels

    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))
    ## the above code will convert shape (14,) to (1, 14)

    return train_X_orig, train_y, test_X_orig, test_y

def explore_dataset(train_X_orig, train_y, test_X_orig, test_y):
    '''
    Function for exploring the dataset.

    Arguments:
    train_X_orig -- our training features
    train_y -- our training labels
    test_X_orig -- our testing features
    test_y -- our testing labels

    Returns --
    <None>
    '''

    m_train = train_X_orig.shape[0]
    num_px = train_X_orig.shape[1]
    m_test = test_X_orig.shape[0]

    print(f'Number of training examples: {m_train}')
    print(f'Number of testing examples: {m_test}')
    print(f'Size of an image: ({num_px}, {num_px})')
    print(f'train_X_orig shape: {train_X_orig.shape}')
    print(f'train_y shape: {train_y.shape}')
    print(f'test_X_orig shape: {test_X_orig.shape}')
    print(f'test_y shape: {test_y.shape}')

def reshapingX(train_X_orig, test_X_orig):
    '''
    Function for reshaping train_X_orig and test_X_orig

    Arguments:
    train_X_orig -- training features with (m, num_px, num_px, 3) shape
    test_X_orig -- testing features with (m, num_px, num_px, 3) shape

    Returns:
    train_X -- training features with (num_px * num_px * 3, m) shape
    test_X -- testing features with (num_px * num_px * 3, m) shape
    '''

    train_X = train_X_orig.reshape(train_X_orig.shape[0], -1).T
    test_X = test_X_orig.reshape(test_X_orig.shape[0], -1).T
    ## the "-1" makes reshape flatten the remaining dimensions

    train_X = train_X / 255
    test_X = test_X / 255
    ## standardize data to have feature values between 0 and 1

    return train_X, test_X

def L_layer_model(X, y, layers_dims, learning_rate=0.0075, num_iterations=3000):
    ''' 
    Implements L layer neural network.

    Arguments:
    X -- training features
    y -- training lables
    layers_dims -- list contaning the input size and each layer size
    learning_rate -- learining rate of gradient descent update rule
    num_iterations -- number of iteration of the optimization loop

    Returns:
    parameters -- parameters learnt by the model which can be used to predict.
    '''

    np.random.seed(1)
    ## setting the seed value to 1
    costs = []

    parameters = network2.initialize_paramters(layers_dims)

    ## Loop(Gradinet Descent)
    for i in range(0, num_iterations):
        AL, caches = network2.L_model_forward(X, parameters)
        ## Forward propagation

        cost = network2.compute_cost(AL, y)
        ## compute cost

        grads = network2.L_layer_backward(AL, y, caches)
        ## backward propagation

        parameters = network2.update_parameters(parameters, grads, learning_rate)
        ## update parameters

        if i % 100 == 0:
            print(f'Cost after {i}th iteration: {cost}')
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds')
    plt.title(f'Learning Rate: {learning_rate}')
    plt.show()

    return parameters

## slothfulwave612