import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def compute_loss(A2, Y):

    one_hot_Y = one_hot(Y)

    log_likelihood = -np.log(A2 + 1e-15)
    loss = np.sum(one_hot_Y * log_likelihood) / Y.size

    return loss

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def gradient_descent(X, Y, alpha, iterations, print_interval=10, save_dir="./original"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    W1, b1, W2, b2 = init_params()
    loss_list = []
    acc_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        loss = compute_loss(A2, Y)
        loss_list.append(loss)

        if i % print_interval == 0:
            predictions = get_predictions(A2)
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

    return W1, b1, W2, b2


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

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500, print_interval=1, save_dir="original")

    ######### test #######

    test_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    test_acc = get_accuracy(test_predictions, Y_dev)
    print("Test Accuracy:", test_acc)

    np.savetxt("original/test_predictions.csv", test_predictions, delimiter=",", fmt="%d")

    with open("original/test_accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc}\n")

