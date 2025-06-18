from mnist_data import load_mnist
from dense import Dense
from activations import *
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime
from network import train, predict
from activations import Softmax
import numpy as np

(x_train, y_train), (x_test, y_test) = load_mnist()

network = [
    Dense(x_train.shape[1], 128),
    Leaky_Relu(),
    Dense(128, 64),
    Leaky_Relu(),
    Dense(64, 10),
    Softmax()
]

train(network, x_train, y_train, cross_entropy, cross_entropy_prime, epochs=100, learning_rate=0.001)

correct_preds = 0
for x,y in zip(x_test, y_test):
    result = predict(network, x)
    if (np.argmax(result)-np.argmax(y)) == 0:
        correct_preds += 1

print(f"Accuracy: {(correct_preds/x_test.shape[0])*100:.2f}%")


import pickle


def save_model(network, filename='model.pkl'):
    data = []
    for layer in network:
        if isinstance(layer, Dense):
            data.append(layer.save())
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

save_model(network, "trained_model.pkl")
