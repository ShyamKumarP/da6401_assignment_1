
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    """Parse command-line arguments (same as train.py)."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, default="mnist",choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",choices=["cross_entropy", "mse", "mean_squared_error"])
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",default=[128, 128, 128])
    parser.add_argument("-a", "--activation", type=str, default="relu",choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",choices=["random", "xavier", "zeros"])

    parser.add_argument("-wp", "--wandb_project", type=str, default='da6401-assignment1')
    parser.add_argument("--model_path", type=str, default="best_model.npy")

    args = parser.parse_args()

    return args



def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data.

    Returns
    -------
    dict with keys: logits, loss, accuracy, f1, precision, recall
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    logits = model.forward(X_test)
    loss      = model.loss_function.forward(logits, y_test)
    y_pred    = np.argmax(logits, axis=1)

    accuracy  = float(np.mean(y_pred == y_test))
    f1        = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    return {
        "logits":    logits,
        "loss":      float(loss),
        "accuracy":  accuracy,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
    }


def main():
    args = parse_arguments()
    _, _, _, _, X_test, y_test = load_data(dataset=args.dataset, val_split=0.1)
    
    best_model = NeuralNetwork(args)
    best_weights = np.load(args.model_path, allow_pickle=True).item()
    best_model.set_weights(best_weights)
    
    results = evaluate_model(best_model, X_test, y_test)

    print("\n=== Evaluation Results ===")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")

    print("\nEvaluation complete!")
    return results


if __name__ == "__main__":
    main()
