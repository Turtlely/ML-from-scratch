import numpy as np

np.random.seed(42)


structure = [784,10]

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def sigmoid_derivative(x):
    return x * (1-x)

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

#print(n.get_params())

input_array = np.random.randn(784)

y_true = np.random.randn(10)

n_epochs = 100
learning_rate = 0.001

# Feedforward propagation
y_pred, node_states = n.predict(input_array)

# Calculate the error
total_error = 0.5*((y_pred - y_true)**2).sum()/len(y_pred)

# Loop through each node of the last layer
for node in range(structure[-1]):
    # Backpropagation
    node_del = -1 * (y_true[node]-y_pred[node]) * y_pred[node] * (1 - y_pred[node])

    # Loop through each node of the next layer. this is to get the output of these nodes and update the connecting weight
    for node_n in range(structure[-2]):
        # Derivative of total error with respect to the output of a certain node
        dE_dwn = n.node_states[-2][node_n]  * node_del

        # Calculate weight adjustment per weight
        update = learning_rate * dE_dwn

        # Update weights as such
        n.weights[-1][node][0,node_n] -= update

        print(n.weights[-1][node][0,node_n])

# Update hidden layer next


# Update weight for neuron in layer "x" number "n"
