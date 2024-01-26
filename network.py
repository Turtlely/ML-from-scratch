import numpy as np

# Random seed
np.random.seed(42)

# 784 inputs and 10 layers
num_Layers = 5
input_dim = 3
structure = [input_dim,num_Layers]

# Sigmoid function for activation
def sigmoid(x):
    return 1/(1+np.exp(-1*x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1-x)

# MSE Cost function, y is predicted and y0 is real
def cost(y,y0):
    return 0.5*(y0-y)**2

class Network:
    # Initialize the network
    def __init__(self,structure):
        self.structure = structure
        self.input_layer = structure[0]
        self.hidden_layers = structure[1:]
        self.node_states = []

        # Generate bias vector per layer
        self.bias = np.array([np.random.randn(n) for n in self.hidden_layers],dtype=object)

        # Generate weights vector per layer
        self.weights = [np.matrix([np.random.randn(self.structure[layer-1]) for node in range(self.structure[layer])]) for layer in  range(len(self.structure))[1:]]

    # Calculate the output of a single layer
    def compute(self,layer, input_array):
        # Work on first layer
        weights = self.weights[layer]
        biases = self.bias[layer]        

        # tracks neuron within the layer
        output_layer = np.array([sigmoid(np.dot(input_array,weights[n].reshape((input_array.shape[0],1)))+biases[n]) for n in range(self.hidden_layers[layer])]).reshape(self.hidden_layers[layer])
        return output_layer

    # Utility function to get network parameters
    def get_params(self):
        print("Weights shape: ",self.weights[-1][0].shape)

    # Generate a prediction from the network using the internal weights and biases recursively
    def predict(self,inp, layer=0):
        # Keep a log of the values of each and every neuron value after prediction

        if layer < len(self.structure)-2:
            out = self.compute(layer,inp)
            self.node_states.append(out)
            return self.predict(out,layer+1)
        else:
            out = self.compute(layer,inp)
            self.node_states.append(out)
            return out.reshape(self.structure[-1],1), self.node_states

n = Network(structure)

n.get_params()

input_array = np.random.randn(input_dim)

y_true = np.random.randn(10)
print("Input")
print(input_array)
print("Bias Matrix")
print(n.bias)
print()
print("Weight Matrix")
print(n.weights)

# Forward Propogation

# Compute weighted inputs for each layer

# Start with input
print("Output of first layer")
#weighted_input_layer=np.array([n.compute(i,input_array)] for i in range(num_Layers))

print(n.compute(1,input_array))

print()


