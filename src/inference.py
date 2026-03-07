"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix 
from ann.neural_network import NeuralNetwork
import ann.objective_functions as obj_funcs
from ann.activations import softmax
import utils.data_loader as data_loader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])

    parser.add_argument('-e', '--epochs', type=int, default=10)

    parser.add_argument('-b', '--batch_size', type=int, default=64)

    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['mean_squared_error', 'cross_entropy'])

    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)

    parser.add_argument('-nhl', '--num_layers', type=int, default=3)

    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128,128,128])

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'])

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier', 'zeros'])

    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401-assignment1')

    parser.add_argument('--model_save_path', type=str,default='best_model.npy')
    return parser.parse_args()

def load_model(model_save_path, args):
    model = NeuralNetwork(args)
    saved_weights = np.load(model_save_path, allow_pickle=True).item()
    model.set_weights(saved_weights)
    return model


def evaluate_model(model, X_test, y_test): 
    logits = model.forward(X_test)
    
    probs = softmax(logits)
    predictions = np.argmax(logits, axis=0)  
    true_labels = np.argmax(y_test, axis=0)
    
    loss = np.mean(obj_funcs.Cross_entropy_forward(y_test, probs)) 

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    conf_mat = confusion_matrix(true_labels, predictions)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_mat
    }

def main():
    args = parse_arguments()
    if args.loss == 'mean_squared_error':
        args.loss = 'mse'
   
    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
    
    _,_,X_test_raw,y_test_raw = data_loader.load_data(args.dataset)
    X_test, y_test = data_loader.pre_processing_data(X_test_raw, y_test_raw)   
    model = load_model(args.model_save_path, args)
    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"Loss:      {results['loss']:.4f}")
    print("Evaluation complete!")

    return results


if __name__ == '__main__':
    main()
