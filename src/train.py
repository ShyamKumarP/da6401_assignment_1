import argparse
import os
import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, pre_processing_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

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

    parser.add_argument('--model_save_path', type=str, default='best_model.npy')

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.loss == 'mean_squared_error':
        args.loss = 'mse'
    
    wandb.init(project=args.wandb_project, config=vars(args))
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(args.dataset)
    
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)

    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
        
    
    model = NeuralNetwork(args=args)
    
    print("Starting training...")
    model.train(
        X_train=X_train, 
        y_train=y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        X_val=X_test, 
        y_val=y_test
    )
    

    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    best_weights = model.get_weights()
    np.save(args.model_save_path, best_weights)

    import json
    with open('src/best_config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    wandb.finish()
    print("Training complete!")

if __name__ == '__main__':
    main()
