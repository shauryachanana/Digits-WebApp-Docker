from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size, weights=None, bias=None):
        if weights is not None and bias is not None:
            self.weights = np.array(weights)
            self.bias = np.array(bias)
        else:
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size) #HE Initialisation to get better variance in the weights for ReLu activation Layer
            self.bias = np.random.randn(output_size, 1)

    def save(self):
        return (self.weights, self.bias)

    def load(self, data):
        self.weights, self.bias = data

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient