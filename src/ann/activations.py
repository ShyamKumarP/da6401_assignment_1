"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

#Linear activation function needed for the output layer, to return logits
class Linear:
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)

#ReLU
class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

#Sigmoid
class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return Sigmoid(x) * (1 - Sigmoid(x))

#Tanh
class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x)**2


Activation_Mapping = {
    "linear": Linear,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh
}

#Getting the required activation function object
def get_activation(name):
    name = name.lower()
    try:
        return Activation_Mapping[name]()
    except KeyError:
        raise ValueError(f"Activation function {name} not found")
