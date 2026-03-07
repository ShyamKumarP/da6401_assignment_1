"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np  

class SGD:
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.grad_W += self.weight_decay * layer.W
            layer.W -= self.lr * layer.grad_W
            layer.bias -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, lr, weight_decay=0.0, momentum=0.9):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = momentum
        self.v_w = {}
        self.v_b = {}

    def update(self, layers):
        for i,layer in enumerate(layers):
            if i not in self.v_w:
                self.v_w[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.bias)
           
            layer.grad_W += self.weight_decay * layer.W
            
            self.v_w[i] = self.lr * layer.grad_W + self.beta * self.v_w[i]
            self.v_b[i] = self.lr * layer.grad_b + self.beta * self.v_b[i]
            
            layer.W -= self.v_w[i]
            layer.bias -= self.v_b[i]

class NAG:
    def __init__(self, lr, weight_decay=0.0, momentum=0.9):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = momentum
        self.v_w = {}
        self.v_b = {}
    def update(self, layers):
        for i,layer in enumerate(layers):
            if i not in self.v_w:
                self.v_w[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.bias)
            layer.grad_W += self.weight_decay * layer.W

            self.v_w[i] = self.lr * layer.grad_W + self.beta * self.v_w[i]
            self.v_b[i] = self.lr * layer.grad_b + self.beta * self.v_b[i]
            layer.W -= (self.beta * self.v_w[i] + self.lr * layer.grad_W)
            layer.bias -= (self.beta * self.v_b[i] + self.lr * layer.grad_b)


class RMSprop:
    def __init__(self, lr=0.01, weight_decay=0.0, decay_rate=0.9,epsilon = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = decay_rate
        self.epsilon = epsilon
        self.v_w = {}
        self.v_b = {}

    def update(self, layers):
        for i,layer in enumerate(layers):
            if i not in self.v_w:
                self.v_w[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.bias)
                
            layer.grad_W += self.weight_decay * layer.W
              
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (layer.grad_b ** 2)
            
            layer.W -= (self.lr / (np.sqrt(self.v_w[i]) + self.epsilon)) * layer.grad_W
            layer.bias -= (self.lr / (np.sqrt(self.v_b[i]) + self.epsilon)) * layer.grad_b
        
