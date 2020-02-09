'''
nn_utils_2.py
-------------

This module contains some helper function which will be used by run_script.py
to make our Neural Network function.

The function here will explore the dataset, reshape it, train the neural
network and test our model.

Function Used(3):
================
1. explore_dataset: for exploring the dataset.
2. reshapingX: for reshaping the shape of X for test and train dataset.
3. two_layer_model: main function where neural network is trained.

Modules Used(4):
===============
1. numpy: package for scientific computing with Python.
2. matplotlib.pyplot: plotting library in Python.
3. network1: provides necessary functions for our neural network. 
4. nn_utils: provides some necessary functions for this module.
'''

import numpy as np
import matplotlib.pyplot as plt
import network1
import nn_utils


def explore_dataset(train_x_orig, train_y, test_x_orig, test_y):
    '''
    This function will be used to explore the dataset we have

    Arguments:
    train_x_orig -- our training features
    train_y -- our training labels
    test_x_orig -- our testing features
    test_y -- our testing labels

    Returns:
    None, only prints the informations about the dataset
    '''    
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    ## explore dataset

    print(f'Number of training examples: {m_train}')
    print(f'Number of testing examples: {m_test}')
    print(f'Size of image: ({num_px}, {num_px}, 3)')
    print(f'train_x_orig shape: {train_x_orig.shape}')
    print(f'train_y shape: {train_y.shape}')
    print(f'test_x_orig shape: {test_x_orig.shape}')
    print(f'test_y shape: {test_y.shape}')

def reshapingX(train_x_orig, test_x_orig):
    '''
    The function will reshape train_x_orig and test_x_orig

    Arguments:
    train_x_orig -- the training dataset features 
    test_x_orig -- the testing dataset features

    Returns:
    train_x -- reshaped train_x_orig 
    test_x -- reshape test_x_orig
    '''

    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    ## the "-1" makes reshape flatten the remaining dimensions

    train_x = train_x / 255
    test_x = test_x / 255
    ## standardize data to have feature values between 0 and 1

    return train_x, test_x

def two_layer_model(X, y, layer_dims, learning_rate, num_iterations, print_cost=False):
    '''    
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data
    y -- true labels
    layer_dims -- dimensions of the layers (n_x, n_h, n_y)
    learning_rate -- learning rate of the gradient descent
    num_iterations -- number of iterations for gradient descent
    print_cost -- if set true, this will print the cost on every 100th iteration

    Returns:
    parameters -- a dictionary containing W1, W2, b1 and b2
    '''    

    np.random.seed(1)
    ## seeting seed value to be 1

    grads = {}
    costs = []
    ## to keep a trak of cost
    (n_x, n_h, n_y) = layer_dims

    parameters = network1.initialize_parameters(n_x, n_h, n_y)
    ## initializing the values of parameters

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    ## initializing W1, b1, W2, b2

    ## Grdient Descent
    for iteration in range(0, num_iterations):
        A1, cache1 = network1.linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = network1.linear_activation_forward(A1, W2, b2, 'sigmoid')
        ## computing A1, A2, cache1 and cache2
        ## linear_activation_forward defined in network1
        
        cost = network1.compute_cost(A2, y)
        ## computing cost
        ## coompute_cost defined in network1

        dA2 = - (np.divide(y, A2) - np.divide(1 - y, 1 - A2))
        ## initializing backward propagation

        dA1, dW2, db2 = network1.linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = network1.linear_activation_backward(dA1, cache1, 'relu')
        ## backward propagation

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        ## setting up grads values

        parameters = network1.update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        ## print the cost after every 100th iterations
        if print_cost and iteration % 100 == 0:
            print(f'Cost after iteration {iteration}: {np.squeeze(cost)}')
        if iteration % 100 == 0:
            costs.append(cost)    

    ## plotting the cost after every 100th iterations
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds)')
    plt.title(f'Learning rate: {learning_rate}')
    plt.show()

    return parameters
    
## slothfulwave612