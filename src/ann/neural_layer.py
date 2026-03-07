"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np  
from ann import activations
class NeuralLayer:
    def __init__(self, input_size, layer_size,weight_init = 'xavier',layer_type = 'hidden',activation_function = 'relu'):
        self.input_size = input_size
        self.layer_size = layer_size
        self.layer_type = layer_type
 
        if weight_init == 'zeros':
            self.W = np.zeros((layer_size, input_size)) 
        elif weight_init == 'random':
            self.W = np.random.randn(layer_size, input_size) * 0.01  
        elif weight_init == 'xavier':
            std_dev = np.sqrt(2 / (self.input_size + self.layer_size))
            self.W = np.random.randn(layer_size, input_size) * std_dev  
        else:
            raise ValueError(f"Unknown")
        
        self.bias = np.zeros((layer_size, 1))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.bias)
        
        if self.layer_type == 'hidden':
            self.activation_function, self.activation_derivative = activations.get_activation(activation_function)
        elif self.layer_type == 'output':
            self.activation_function, self.activation_derivative = activations.get_activation('softmax')
        
    
    def forward(self, input_vector):
        self.input_vector = input_vector  
        self.weighted_sum = np.dot(self.W, input_vector) + self.bias  
        self.output = self.activation_function(self.weighted_sum)  
        return self.output
    

    def backward(self, dL_dz_next):
        dL_dz = dL_dz_next * self.activation_derivative(self.weighted_sum)

        self.grad_W = np.dot(dL_dz,self.input_vector.T) / self.input_vector.shape[1]  
        self.grad_b = np.sum(dL_dz, axis=1, keepdims=True) / self.input_vector.shape[1]  
        
        dL_dz_prev = np.dot((self.W).T,dL_dz) 
        return dL_dz_prev
