"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from ann.activations import get_activation


class NeuralLayer:
    """
    intput size : no of neurons in the previous layer
    output size : no of neurons in the current layer
    activation : activation used in current layer
    weights_init : how weights are initialized in current layer
    """
    def __init__(self, input_dim, output_dim, activation = "relu", weight_init = "xavier"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.activation = get_activation(activation)

        
        self.W, self.b = self._weight_initialization(weight_init)

        
        self.z = None       
        self.input_cache = None    

        self.grad_W = None
        self.grad_b = None

    

    def _weight_initialization(self, weight_init):
        weight_init = weight_init.lower()
        if weight_init == "zeros":
            W = np.zeros((self.input_dim, self.output_dim))
            b = np.zeros((1, self.output_dim))
        elif weight_init == "random":
            W = np.random.randn(self.input_dim, self.output_dim) * 0.01
            b = np.zeros((1, self.output_dim))
        elif weight_init == "xavier":
            limit = np.sqrt(4.0 / (self.input_dim + self.output_dim))
            W = np.random.uniform(-limit, limit, (self.input_dim, self.output_dim))
            b = np.zeros((1, self.output_dim))
        else:
            raise ValueError(f"Unknown weight_init '{weight_init}'")
        return W, b

    

    def forward(self, input_cache):
        self.input_cache = input_cache                          
        self.z = input_cache @ self.W + self.b          
        return self.activation.forward(self.z)

    

    def backward(self, delta):  
        delta_z = delta * self.activation.backward(self.z)                     

        batch_size = self.input_cache.shape[0]
        
        self.grad_W = (self.input_cache.T @ delta_z) / batch_size   
        self.grad_b = np.sum(delta_z, axis=0, keepdims=True) / batch_size  

        delta_prev = delta_z @ self.W.T                    
        return delta_prev

