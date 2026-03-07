"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
import wandb
from sklearn.metrics import f1_score
from ann import optimizers,objective_functions
from ann.neural_layer import NeuralLayer
from utils.data_loader import batch_generator


class NeuralNetwork:
    def __init__(self, args):
        self.layers = []
        self.num_layers = args.num_layers
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.layers_sizes = args.hidden_size
        self.weight_init = args.weight_init
        self.activation = args.activation
        self.loss_type = args.loss

        self.input_dim = 784  
        self.output_dim = 10
        
        # Optimizer
        opt_map = {
            "sgd": optimizers.SGD,
            "momentum": optimizers.Momentum,
            "nag": optimizers.NAG,
            "rmsprop": optimizers.RMSprop
        }
        self.optimizer = opt_map[args.optimizer](lr=self.lr, weight_decay=self.weight_decay)
        
        prev_dim = self.input_dim

        for size in self.layers_sizes[:self.num_layers]:
            self.layers.append(NeuralLayer(input_size=prev_dim,layer_size=size,weight_init=self.weight_init,layer_type="hidden",activation_function=self.activation))
            prev_dim = size
        self.layers.append(NeuralLayer(input_size=prev_dim,layer_size=self.output_dim,weight_init=self.weight_init,layer_type="output"))
    

    def forward(self, X):
        signal = X
        for layer in self.layers:
            signal = layer.forward(signal)
        return self.layers[-1].weighted_sum
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        probs = self.layers[-1].output

        if self.loss_type == "cross_entropy":
            grad = probs - y_true
        else:
            dL_da = objective_functions.Mean_Square_backward(y_true, probs)
            temp = np.sum(probs * dL_da, axis=0, keepdims=True)
            grad = probs * (dL_da - temp)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)
        
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b
        
    def step(self):
        self.optimizer.update(self.layers)

    def get_weights(self):
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f'W{i}'] = layer.W.copy()
            weights_dict[f'layer_{i}_bias'] = layer.bias.copy()
        return weights_dict
        
    def set_weights(self, weights_dict):
        for i, layer in enumerate(self.layers):
            if f'W{i}' in weights_dict and f'layer_{i}_bias' in weights_dict:
                layer.W = weights_dict[f'W{i}'].copy()
                layer.bias = weights_dict[f'layer_{i}_bias'].copy()
           
    def train(self, X_train, y_train, epochs=1, batch_size=64, X_val=None, y_val=None):

        best_f1 = -1
        best_weights = self.get_weights()

        epoch_losses = []

        for epoch in range(epochs):

            correct_predictions = 0
            batch_losses = []

            for X_batch, y_batch in batch_generator(X_train, y_train, batch_size):

                logits = self.forward(X_batch)

                predicted = np.argmax(logits, axis=0)
                actual = np.argmax(y_batch, axis=0)

                correct_predictions += np.sum(predicted == actual)

                probs = self.layers[-1].output

                if self.loss_type == "cross_entropy":
                    loss = objective_functions.Cross_entropy_forward(y_batch, probs)
                else:
                    loss = objective_functions.Mean_Square_forward(y_batch, probs)

                batch_losses.append(np.mean(loss))

                self.backward(y_batch, logits)
                self.step()

            train_accuracy = correct_predictions / X_train.shape[1]
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_acc, val_loss, val_f1 = self.evaluate(X_val, y_val, verbose=False)

                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss {epoch_loss:.4f} | "
                    f"Train Acc {train_accuracy:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}"
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_weights = self.get_weights()

                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1
                })

            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss {epoch_loss:.4f} | Train Acc {train_accuracy:.4f}")
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": train_accuracy
                })

        self.set_weights(best_weights)
        return epoch_losses

    def evaluate(self, X, y, verbose=True):
        logits = self.forward(X)

        predicted = np.argmax(logits, axis=0)
        actual = np.argmax(y, axis=0)

        accuracy = np.mean(predicted == actual)
        f1 = f1_score(actual, predicted, average="macro")

        probs = self.layers[-1].output

        if self.loss_type == "cross_entropy":
            loss = np.mean(objective_functions.Cross_entropy_forward(y, probs))
        else:
            loss = np.mean(objective_functions.Mean_Square_forward(y, probs))
        if verbose:
            print(f"Accuracy: {accuracy}")
            print(f"F1 Score: {f1}")
            print(f"Loss: {loss}")
        return accuracy, loss, f1
