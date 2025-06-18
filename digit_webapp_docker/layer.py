class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        #overriden functions in child classes
        pass

    def backward(self, output_gradient, learning_rate):
        #overriden functions in child classes
        pass