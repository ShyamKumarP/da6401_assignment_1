"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
def Cross_entropy_forward(y_true, y_probs):
    probs_clipped = np.clip(y_probs, 1e-12, 1.0 - 1e-12)
    loss = -np.sum(y_true * np.log(probs_clipped), axis=0) 
    return loss 

def Cross_entropy_backward(y_true, y_probs):
    return y_probs - y_true 

def Mean_Square_forward(y_true, y_probs):
    loss = np.mean((y_true - y_probs)**2, axis=0)
    return loss 

def Mean_Square_backward(y_true, y_probs):
    return 2*(y_probs - y_true)/y_true.shape[0]
