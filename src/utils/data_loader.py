import numpy as np
import wandb

FASHION_MNIST_LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

MNIST_LABELS = [str(i) for i in range(10)]


def _load_keras_dataset(name):
    #Load raw arrays from keras.datasets

    if name == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return X_train, y_train, X_test, y_test


def _preprocess(X: np.ndarray) -> np.ndarray:
    #Flatten (N,28,28) -> (N,784) and normalise to [0,1]
    return X.reshape(X.shape[0], -1).astype(np.float64) / 255.0


def load_data(dataset = "mnist",val_split = 0.1,seed = 69):
   
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = _load_keras_dataset(dataset)

    
    X_train_dataset = _preprocess(X_train_raw)
    y_train_dataset = y_train_raw
    X_test = _preprocess(X_test_raw)
    y_test = y_test_raw

    # Train / validation split
    rng = np.random.default_rng(seed)
    n = X_train_dataset.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_split)

    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X_train_dataset[train_idx], y_train_dataset[train_idx]
    X_val,   y_val   = X_train_dataset[val_idx],   y_train_dataset[val_idx]

    print(f"Loaded {dataset}: "
          f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test
