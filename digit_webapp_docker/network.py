def predict(network, input):
    for layer in network:
        output = layer.forward(input)
        input = output
    return output;

def train(network, x_train, y_train, loss, loss_prime, epochs=10, learning_rate = 0.01, verbose=True):
    for e in range(epochs):
        error = 0

        for x,y in zip(x_train, y_train):
            input = x
            for layer in network:
                output = layer.forward(input)
                input = output
            
            error += loss(y, output)
            grad = loss_prime(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(x_train) #average error over the number of samples
        if(verbose):
                print(f"Error at Epoch #{e+1}/{epochs}: {error:.6f}")
            

            
