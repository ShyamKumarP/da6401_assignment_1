"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


def _softmax(logits):
    exp_z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _one_hot(y, num_classes):
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1.0
    return one_hot


class CrossEntropy:
    def forward(self, logits, y_true):
        probs = _softmax(logits)
        n = logits.shape[0]
        log_probs = np.log(np.clip(probs[np.arange(n), y_true.astype(int)], 1e-12, 1.0 - 1e-12))
        return -np.mean(log_probs)

    def backward(self, logits, y_true):
        probs = _softmax(logits)
        one_hot_matrix = _one_hot(y_true, logits.shape[1])
        
        return (probs - one_hot_matrix)


class MeanSquaredError:
    def forward(self, logits, y_true):
        probs = _softmax(logits)
        one_hot_matrix = _one_hot(y_true, logits.shape[1])
        return np.mean(np.sum((probs - one_hot_matrix) ** 2, axis=1))

    def backward(self, logits, y_true):
        
        probs = _softmax(logits)
        one_hot_matrix = _one_hot(y_true, logits.shape[1])


        weighted = np.sum((probs - one_hot_matrix) * probs, axis=1, keepdims=True)
        
        grad = 2.0 * probs * ((probs - one_hot_matrix) - weighted)
        return grad



Loss_Function_mapping = {
    "cross_entropy": CrossEntropy,
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
}


def get_loss_function(name: str):
    name = name.lower()
    try:
        return Loss_Function_mapping[name]()
    except KeyError:
        raise ValueError(f"Unknown loss '{name}'")
