import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data(dataset):
    cache_prefix = f"src/{dataset}"

    if os.path.exists(f"{cache_prefix}_X_train.npy"):
        X_tr = np.load(f"{cache_prefix}_X_train.npy", mmap_mode="r")
        y_tr = np.load(f"{cache_prefix}_y_train.npy", mmap_mode="r")
        X_te = np.load(f"{cache_prefix}_X_test.npy", mmap_mode="r")
        y_te = np.load(f"{cache_prefix}_y_test.npy", mmap_mode="r")

        return X_tr, y_tr, X_te, y_te
    
    if dataset == "mnist":
        data = fetch_openml("mnist_784", version=1, parser="liac-arff", as_frame=False)

    elif dataset == "fashion_mnist":
        data = fetch_openml("Fashion-MNIST", version=1, parser="liac-arff", as_frame=False)

    X = data.data.astype(np.float32)
    y = data.target.astype(np.uint8)

    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)
    
    np.save(f"src/{dataset_name}_X_train.npy", X_tr)
    np.save(f"src/{dataset_name}_y_train.npy", y_tr)
    np.save(f"src/{dataset_name}_X_test.npy", X_te)
    np.save(f"src/{dataset_name}_y_test.npy", y_te)

    return X_tr, y_tr, X_te, y_te

def pre_processing_data(X,y):
    X_norm = X.astype(np.float32) / 255.0
    one_hot = np.eye(10)[y]
    return X_norm.T, one_hot.T

def batch_generator(X,y,batch_size):
    n_samples = X.shape[1]
    order = np.random.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        idx = order[start:end]
        yield X[:, idx], y[:, idx]
