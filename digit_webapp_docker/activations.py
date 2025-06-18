from activation import Activation
from layer import Layer
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class ReLu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x>0, 1, 0)
        super().__init__(relu, relu_prime)
        
class Leaky_Relu(Activation):
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
        leaky_relu = lambda x: np.where(x >= 0, x, self.alpha*x)
        leaky_relu_prime = lambda x: np.where(x >= 0, 1, self.alpha)
        super().__init__(leaky_relu, leaky_relu_prime)
        
class Softmax(Layer):
    def forward(self, input):
        self.input = input
        exps = np.exp(input - np.max(input))
        return exps / np.sum(exps)
    def backward(self, output_gradient, learning_rate):
        return output_gradient # works because we are using corss entropy loss