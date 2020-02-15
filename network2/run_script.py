'''
run_script.py
-------------

The module will let the users run our neural network.

Module Used(5):
1. numpy: package for scientific computing with Python.
2. matplotlib.pyplot: plotting library.
3. network1: provides necessary functions for our neural network.
4. nn_utils: provides necessary function for this module.
5. nn_utils_2: provides some necessary functions for this module.
'''

import numpy as np
import matplotlib.pyplot as plt
import network2
import nn_utils
import nn_utils_2

train_X_orig, train_y, test_X_orig, test_y = nn_utils_2.load_data()
## load_data function is defined in nn_utils_2
## used for loading up the data

## exploring the dataset
print('\n***Exploring the Dataset***\n')
nn_utils_2.explore_dataset(train_X_orig, train_y, test_X_orig, test_y)

print()
input('Press Enter To Continue')
print()
## reshaping the train_X_orig and test_X_orig
## train_X_orig and test_X_orig is of the shape (m, num_px, num_px, 3)
## the required shape is (num_px * num_px * 3, m)
train_X, test_X = nn_utils_2.reshapingX(train_X_orig, test_X_orig)

## hyperparameters
n_x = 12288             ## num_px * num_px * 3
n_h = [20, 7, 5]        ## 3 hidden layers with 20, 7 and 5 neurons respectively
n_y = 1                 ## output layer having one neuron

layers_dims = [n_x] 
layers_dims.extend(n_h)
layers_dims.append(n_y)

parameters = nn_utils_2.L_layer_model(train_X, train_y, layers_dims)
## model training
input('\nPress Enter To Continue')
print()

print('Training Accuracy:')
network2.predict(train_X, train_y, parameters)

print()
print('Test Accuracy:')
network2.predict(test_X, test_y, parameters)

## slothfulwave612