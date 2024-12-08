import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

def init_params():

    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    W3 = np.random.rand(10, 64) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, X):

    Z1 = W1.dot(X) + b1   # Z1: [128, m]
    A1 = ReLU(Z1)         # A1: [128, m]

    Z2 = W2.dot(A1) + b2  # Z2: [64, m]
    A2 = ReLU(Z2)         # A2: [64, m]

    Z3 = W3.dot(A2) + b3  # Z3: [10, m]
    A3 = softmax(Z3)      # A3: [10, m]

    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def compute_loss(A3, Y):
    one_hot_Y = one_hot(Y)
    log_likelihood = -np.log(A3 + 1e-15)
    loss = np.sum(one_hot_Y * log_likelihood) / Y.size
    return loss

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)


    dZ3 = A3 - one_hot_Y              # [10, m]
    dW3 = (1/m) * dZ3.dot(A2.T)       # [10,64]
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)  # [10,1]

    dA2 = W3.T.dot(dZ3)               # [64,m]
    dZ2 = dA2 * ReLU_deriv(Z2)        # [64,m]
    dW2 = (1/m) * dZ2.dot(A1.T)       # [64,128]
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # [64,1]

    dA1 = W2.T.dot(dZ2)               # [128,m]
    dZ1 = dA1 * ReLU_deriv(Z1)        # [128,m]
    dW1 = (1/m) * dZ1.dot(X.T)        # [128,784]
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # [128,1]

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3,
                  dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def gradient_descent(X, Y, alpha, iterations, print_interval=10, save_dir="./extend"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    W1, b1, W2, b2, W3, b3 = init_params()
    loss_list = []
    acc_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        loss = compute_loss(A3, Y)
        loss_list.append(loss)

        if i % print_interval == 0:
            predictions = get_predictions(A3)
            acc = get_accuracy(predictions, Y)
            acc_list.append(acc)
            print("Iteration:", i, "Loss:", loss, "Accuracy:", acc)

    plt.figure()
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    iterations_recorded = range(0, iterations, print_interval)
    plt.figure()
    plt.plot(iterations_recorded, acc_list, label='Training Accuracy')
    plt.xlabel(f"Iterations (every {print_interval})")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
    plt.close()

    return W1, b1, W2, b2, W3, b3


if __name__ == '__main__':

    data = pd.read_csv('./train/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _, m_train = X_train.shape

    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.10, 500, print_interval=1, save_dir="extend")


    test_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
    test_acc = get_accuracy(test_predictions, Y_dev)
    print("Test Accuracy:", test_acc)

    np.savetxt("extend/test_predictions.csv", test_predictions, delimiter=",", fmt="%d")
    with open("extend/test_accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc}\n")
