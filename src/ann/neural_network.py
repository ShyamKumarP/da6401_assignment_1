import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss_function
from ann.optimizers import get_optimizer


def _softmax(logits):
    exp_z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class NeuralNetwork:
    
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers: list = []
       
        input_dim   = 784       # 28*28 input size for MNIST / Fashion-MNIST
        num_classes = 10

        hidden_sizes = cli_args.hidden_size  
        activation   = cli_args.activation   
        weight_init  = cli_args.weight_init

        prev_dim = input_dim
        for h_size in hidden_sizes:
            self.layers.append(NeuralLayer(prev_dim, h_size, activation=activation, weight_init=weight_init))
            prev_dim = h_size

        self.layers.append(NeuralLayer(prev_dim, num_classes, activation="linear", weight_init=weight_init))    # output layer and without softmax

        self.loss_function = get_loss_function(cli_args.loss)

    
        opt_name = cli_args.optimizer.lower()
        lr       = cli_args.learning_rate
        if opt_name in ("momentum", "nag"):
            self.optimizer = get_optimizer(opt_name, learning_rate=lr)
        elif opt_name == "rmsprop":
            self.optimizer = get_optimizer(opt_name, learning_rate=lr)
        elif opt_name in ("adam", "nadam"):
            self.optimizer = get_optimizer(opt_name, learning_rate=lr)
        else:
            self.optimizer = get_optimizer(opt_name, learning_rate=lr)

        self.weight_decay = cli_args.weight_decay

        # Stored grad while backward()
        self.grad_W = None
        self.grad_b = None

   
    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a


    def backward(self, y_true, y_pred):
        
        delta = self.loss_function.backward(y_pred, y_true)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    

    def train(self, X_train, y_train, X_val = None, y_val = None, wandb_run = None):
        
        epochs     = self.args.epochs
        batch_size = self.args.batch_size
        n          = X_train.shape[0]
        history    = {"train_loss": [], "train_acc": [], "val_loss":   [], "val_acc":   []}

        use_nag = self.args.optimizer.lower() == "nag"

        for epoch in range(epochs):
            idx            = np.random.permutation(n)
            X_sample, y_sample = X_train[idx], y_train[idx]
            epoch_loss     = 0.0
            correct        = 0

            for start in range(0, n, batch_size):
                Xb = X_sample[start: start + batch_size]
                yb = y_sample[start: start + batch_size]

                if use_nag:
                    self.optimizer.apply_lookahead(self.layers)

                logits = self.forward(Xb)

                if use_nag:
                    self.optimizer.undo_lookahead(self.layers)

                self.backward(yb, logits)
                self.optimizer.step(self.layers, weight_decay=self.weight_decay)

                epoch_loss += self.loss_function.forward(logits, yb) * Xb.shape[0]
                correct    += int(np.sum(np.argmax(logits, axis=1) == yb))

            epoch_loss /= n
            epoch_acc   = correct / n
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_loss = self.loss_function.forward(val_logits, y_val)
                val_acc  = float(np.mean(np.argmax(val_logits, axis=1) == y_val))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # W&B
            if wandb_run is not None:
                log_dict = {
                    "epoch":      epoch + 1,
                    "train_loss": epoch_loss,
                    "train_acc":  epoch_acc,
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                    log_dict["val_acc"]  = val_acc

                wandb_run.log(log_dict)

            val_str = (f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                       if val_loss is not None else "")
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"train_loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}{val_str}")

        return history

   

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss   = self.loss_function.forward(logits, y)
        accuracy  = float(np.mean(np.argmax(logits, axis=1) == y))
        return loss, accuracy, logits

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
