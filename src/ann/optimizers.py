import numpy as np


class BaseOptimizer:
    """
    Base optimizer class.
    """
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, layers, weight_decay = 0.0):
        raise NotImplementedError

    def _apply_weight_decay(self, layers, weight_decay):
        if weight_decay > 0:
            for layer in layers:
                layer.grad_W = layer.grad_W + weight_decay * layer.W
        return layers



class SGD(BaseOptimizer):
    """
    Mini-batch Stochastic Gradient Descent.
        W <- W - lr * (dL/dW + wd * W)
        b <- b - lr * dL/db
    """

    def __init__(self, learning_rate = 0.01):
        super().__init__(learning_rate)

    def step(self, layers, weight_decay = 0.0):
        self._apply_weight_decay(layers, weight_decay)
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b




class Momentum(BaseOptimizer):
    """
    SGD with classical momentum.
        v_W <- beta * v_W + dL/dW
        W   <- W - lr * v_W
    """

    def __init__(self, learning_rate = 0.01, beta = 0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W: dict = {}
        self.v_b: dict = {}

    def _init_velocities(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def step(self, layers, weight_decay = 0.0):
        self._init_velocities(layers)
        self._apply_weight_decay(layers, weight_decay)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]





class NAG(BaseOptimizer):
    """
    SGD with Nesterov-accelerated gradient.
        v_W <- beta * v_W + dL/dW
        W   <- W - lr * (dL/dW + beta * v_W)
    """

    def __init__(self, learning_rate = 0.01, beta = 0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W: dict = {}
        self.v_b: dict = {}

    def _init_velocities(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def apply_lookahead(self, layers):
        
        self._init_velocities(layers)
        for i, layer in enumerate(layers):
            layer.W -= self.lr * self.beta * self.v_W[i]
            layer.b -= self.lr * self.beta * self.v_b[i]

    def undo_lookahead(self, layers):
        
        for i, layer in enumerate(layers):
            layer.W += self.lr * self.beta * self.v_W[i]
            layer.b += self.lr * self.beta * self.v_b[i]

    def step(self, layers, weight_decay = 0.0):
        self._init_velocities(layers)
        self._apply_weight_decay(layers, weight_decay)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]



class RMSProp(BaseOptimizer):
    """
    RMSProp optimizer.
        s_W <- rho * s_W + (1 - rho) * (dL/dW)^2
        W   <- W - lr * dL/dW / sqrt(s_W + epsilon)
    """
    def __init__(self, learning_rate = 0.001,
                 rho = 0.9, epsilon = 1e-10):
        super().__init__(learning_rate)
        self.rho     = rho
        self.epsilon = epsilon
        self.s_W: dict = {}
        self.s_b: dict = {}

    def _init_cache(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.s_W:
                self.s_W[i] = np.zeros_like(layer.W)
                self.s_b[i] = np.zeros_like(layer.b)

    def step(self, layers, weight_decay = 0.0):
        self._init_cache(layers)
        self._apply_weight_decay(layers, weight_decay)
        for i, layer in enumerate(layers):
            self.s_W[i] = (self.rho * self.s_W[i] +
                           (1 - self.rho) * layer.grad_W ** 2)
            self.s_b[i] = (self.rho * self.s_b[i] +
                           (1 - self.rho) * layer.grad_b ** 2)
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.epsilon)




OPTIMIZER_MAP = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp
}


def get_optimizer(name, learning_rate):
    name = name.lower()
    try:
        return OPTIMIZER_MAP[name](learning_rate)
    except KeyError:
        raise ValueError(f"Unknown optimizer '{name}'")
