from itertools import product
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

    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'])

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier', 'zeros'])

    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401-assignment1')

    parser.add_argument('--model_save_path', type=str, default='src/best_model.npy')

    return parser.parse_args()


def main():

    base_args = parse_arguments()

    # Search space
    optimizers = ['sgd', 'momentum', 'nag', 'rmsprop']
    activations = ['relu', 'sigmoid', 'tanh']
    hidden_sizes = [[64,64,64], [128,128,128], [256,128,64]]
    learning_rates = [0.0005, 0.001, 0.005]
    weight_inits = ['random', 'xavier', 'zeros']

    best_acc = -1
    best_weights = None
    best_config = None

    print("Loading dataset...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(base_args.dataset)

    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)

    wandb.init(project="da6401-assignment1",group="hyperparameter_search")
    run_id = 0

    for opt in optimizers:
        for act in activations:
            for hs in hidden_sizes:
                for lr in learning_rates:
                    for w_init in weight_inits:
                        run_id += 1

                        args = argparse.Namespace(**vars(base_args))

                        args.optimizer = opt
                        args.activation = act
                        args.hidden_size = hs
                        args.learning_rate = lr
                        args.weight_init = w_init
                        args.num_layers = len(hs)

                        if args.loss == 'mean_squared_error':
                            args.loss = 'mse'

                        print("\nRunning experiment with:")
                        print(vars(args))

                    
                        model = NeuralNetwork(args=args)

                        model.train(
                            X_train=X_train,
                            y_train=y_train,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            X_val=X_test,
                            y_val=y_test
                        )

                        acc, loss, f1 = model.evaluate(X_test, y_test, verbose=False)

                        print("Validation accuracy:", acc)

                        wandb.log({
                        "experiment_id": run_id,
                        "optimizer": opt,
                        "activation": act,
                        "hidden_size": str(hs),
                        "learning_rate": lr,
                        "weight_init": w_init,
                        "val_accuracy": acc,
                        "val_loss": loss,
                        "val_f1": f1
                    })
                        if acc > best_acc:
                            best_acc = acc
                            best_weights = model.get_weights()
                            best_config = vars(args)

                       

    print("\nBest Accuracy:", best_acc)

    np.save(base_args.model_save_path, best_weights)

    import json
    with open('src/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=4)

    print("Best model and config saved.")
    wandb.finish()



if __name__ == '__main__':

    main()
