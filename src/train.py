import argparse
import json
import os, sys
import numpy as np

# Allow running from the src/ directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    """Parse command-line arguments."""
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
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Model saving
    parser.add_argument("--model_save_path", type=str, default="best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="best_config.json")
    parser.add_argument("--log_gradients", action="store_true")
    
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--log_class_samples", action="store_true")

    args = parser.parse_args()

    return args


def save_model(model, config, model_path, config_path):
    best_weights = model.get_weights()
    np.save(model_path, best_weights)

    best_config = {
        "dataset":      config.dataset,
        "epochs":       config.epochs,
        "batch_size":   config.batch_size,
        "loss":         config.loss,
        "optimizer":    config.optimizer,
        "weight_decay": config.weight_decay,
        "learning_rate": config.learning_rate,
        "num_layers":   config.num_layers,
        "hidden_size":  config.hidden_size,
        "activation":   config.activation,
        "weight_init":  config.weight_init,
    }
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Model config saved in '{config_path}'")


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                reinit=True,
            )
        except Exception as e:
            print(f"[WARNING] Could not initialise W&B: {e}")

   
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset=args.dataset, val_split=args.val_split, seed=args.seed)


    model = NeuralNetwork(args)

    
    history = model.train(X_train, y_train,X_val=X_val, y_val=y_val,wandb_run=wandb_run)

    from sklearn.metrics import f1_score, precision_score, recall_score
    test_loss, test_acc, test_logits = model.evaluate(X_test, y_test)
    y_pred_labels = np.argmax(test_logits, axis=1)

    f1        = f1_score(y_test, y_pred_labels, average="macro", zero_division=0)
    precision = precision_score(y_test, y_pred_labels, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred_labels, average="macro", zero_division=0)
    if wandb_run:
        wandb_run.log({
            "test_acc":       test_acc,
            "test_f1":        f1,
            "test_precision": precision,
            "test_recall":    recall,
        })

    
    save_model(model, args, args.model_save_path, args.config_save_path)

    if wandb_run:
        wandb_run.finish()

    print("\nTraining complete!")
    return history


if __name__ == "__main__":

    main()
