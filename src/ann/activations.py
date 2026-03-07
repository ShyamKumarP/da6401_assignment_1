"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x) 
def relu_derivative(x):
    return (x > 0).astype(float) 

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return (1 - tanh(x)**2)

def softmax(x):
    #for stability
    shifted_x = x - np.max(x, axis = 0, keepdims=True)
    exp_k = np.exp(shifted_x)
    s = exp_k/np.sum(exp_k, axis=0, keepdims=True)
    return s 
def softmax_derivative(x):
    s = softmax(x)
    return np.ones_like(x)

ACTIVATION_MAPPING = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative),
}

def get_activation(name):
    return ACTIVATION_MAPPING[name]
