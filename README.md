# DA6401 Assignment 1

## Submission Link
Github Link - https://github.com/ShyamKumarP/da6401_assignment_1.git
Weights & Biases Report - https://wandb.ai/me22b223-iitm/da6401-assignment1/reports/gxhxfhgfxhgfghxdh--VmlldzoxNjEzNjE2OQ?accessToken=aqqkqeo68xmr6m1b5hb0s3q9kkxkre9r1wbmxwm40aqdz43j1kszdb8fq96uv1tv

Shyam Kumar- ME22B223

## Neural Network Implementation from Scratch (NumPy)

This project implements a Fully Connected Neural Network from scratch using NumPy for image classification on the MNIST and Fashion-MNIST datasets.

The goal of this assignment is to understand how neural networks work internally by implementing all components manually without using deep learning frameworks like TensorFlow or PyTorch.

The implementation includes:

- Forward Propagation
- Backpropagation
- Activation Functions
- Loss Functions
- Optimizers
- Mini-batch Gradient Descent
- Experiment Tracking using Weights & Biases (WandB)

------------------------------------------------------------

PROJECT PIPELINE

Load Dataset
↓
Preprocess Data
↓
One-Hot Encode Labels
↓
Initialize Neural Network
↓
Forward Propagation
↓
Loss Computation
↓
Backpropagation
↓
Optimizer Updates
↓
Training Loop
↓
Model Saving

------------------------------------------------------------

DATASET

The project supports the following datasets.

MNIST
- 60,000 training images
- 10,000 test images
- Image size: 28 × 28
- Classes: 10 (digits 0–9)

Fashion-MNIST
- 60,000 training images
- 10,000 test images
- Image size: 28 × 28
- Classes: 10 clothing categories

Datasets are downloaded using OpenML.

------------------------------------------------------------

DATA PREPROCESSING

1. Normalization

Pixel values range from:

0 – 255

These are normalized to:

0 – 1

using

X = X / 255.0

This improves training stability and gradient convergence.

------------------------------------------------------------

2. One-Hot Encoding

Labels are converted to one-hot vectors.

Example

Label = 3

One-hot = [0 0 0 1 0 0 0 0 0 0]

------------------------------------------------------------

NEURAL NETWORK ARCHITECTURE

The neural network consists of multiple fully connected layers.

Each layer performs

Z = W · X + b
A = activation(Z)

Where

W = Weight matrix
b = Bias vector
Z = Linear output
A = Activation output

------------------------------------------------------------

ACTIVATION FUNCTIONS

Sigmoid

σ(x) = 1 / (1 + e^-x)

Tanh

tanh(x)

ReLU

max(0, x)

Softmax

softmax(z_i) = exp(z_i) / Σ exp(z_j)

Used in the output layer for multi-class classification.

------------------------------------------------------------

BACKPROPAGATION

Gradients are computed using the chain rule.

dZ = dA * activation_derivative(Z)
dW = (1/m) * dZ · A_prev^T
db = (1/m) * Σ dZ

These gradients are used to update model parameters.

------------------------------------------------------------

OPTIMIZERS IMPLEMENTED

SGD (Stochastic Gradient Descent)

W = W - lr * dW

Momentum

Accelerates gradient descent using velocity terms.

NDG & RMSprop

------------------------------------------------------------

TRAINING PROCEDURE

Training is done using mini-batch gradient descent.

For each epoch:

1. Shuffle training data
2. Split into mini-batches
3. Forward propagation
4. Compute loss
5. Backpropagation
6. Update parameters using optimizer
7. Log metrics using WandB

------------------------------------------------------------

EXPERIMENT TRACKING

Weights & Biases (WandB) is used for experiment tracking.

Tracked metrics include:

- Training loss
- Validation loss
- Accuracy
- Hyperparameters

------------------------------------------------------------
INSTALLATION

Install dependencies

pip install -r requirements.txt

------------------------------------------------------------

COMMAND LINE ARGUMENTS

--dataset           Dataset (mnist / fashion_mnist)
--epochs            Number of training epochs
--batch_size        Batch size
--learning_rate     Learning rate
--optimizer         Optimizer type
--wandb_project     WandB project name
--model_save_path   Path to save trained model

-----------------------------------------------------------

KEY LEARNING OUTCOMES

This project demonstrates understanding of

- Neural network architecture
- Forward propagation
- Backpropagation
- Gradient-based optimization
- Activation functions
- Loss functions
- Data preprocessing
- Experiment tracking

Implementing neural networks from scratch provides deeper insight into how modern deep learning frameworks operate internally.

------------------------------------------------------------

AUTHOR

Shyam Kumar
M.Tech Quantitative Finance
IIT Madras
