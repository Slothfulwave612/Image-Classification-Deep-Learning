'''
run_script.py
-------------

The module will let the users run our neural network.

Module Used(4):
1. numpy: package for scientific computing with Python.
2. matplotlib.pyplot: plotting library.
3. network1: provides necessary functions for our neural network.
4. nn_utils: provides necessary function for this module.
5. nn_utils_2: provides some necessary functions for this module.
'''

import numpy as np
import matplotlib.pyplot as plt
import network1
import nn_utils
import nn_utils_2

train_x_orig, train_y, test_x_orig, test_y = nn_utils.load_data()
## load_data defined in nn_utils

### Let's first explore the dataset
print(f'\n***Exploring The Dataset***\n')
nn_utils_2.explore_dataset(train_x_orig, train_y, test_x_orig, test_y)

print()
input('Press Enter To Continue')
print()

## reshaping the train_x_orig and test_x_orig
## train_x_orig and test_x_orig is in shape (m,num_px,num_px,3)
## the required shape is (num_px*num_px*3, m)
train_x, test_x = nn_utils_2.reshapingX(train_x_orig, test_x_orig)

## hyperparameters
n_x = 12288     # num_px * num_px * 3
n_h = 7         # number of hidden layer nodes
n_y = 1         # number of output layer nodes
layers_dims = (n_x, n_h, n_y)

print_cost = True
if print_cost:
    print()
    print('Cost after every 100th iteration:')
    print()

parameters = nn_utils_2.two_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=print_cost)
## model training

print()
input('Press Enter To Continue')
print()

network1.predict(train_x, train_y, parameters, accuracy='Accuracy Train')
## training accuracy

print()

network1.predict(test_x, test_y, parameters, accuracy='Accuracy Test')

## slothfulwave612