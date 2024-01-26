import numpy as np

np.random.seed(42)

# Network Dimensions
num_layers = 1
input_dim = 1

# Input to the network
#input_array = np.random.randn(input_dim,1)
input_array = np.asarray([[0]])

# True output
#true_output = np.random.randn(input_dim,1)
true_output = np.asarray([[0]])

learning_rate = 10

def activation(x):
    return 1/(1+np.exp(-1*x))

# Derivative of sigmoid function
def activation_derivative(x):
    return x * (1-x)

def cost(x):
    return 1/2 * (x-true_output)**2

def cost_derivative(x):
    return (x-true_output)

class Network:
    def __init__(self,num_layers,input_dim):
        # Bias
        self.bias = np.asarray([np.zeros((input_dim,1)) for i in range(num_layers)])
        # Weights
        self.weights = np.asarray([np.random.randn(input_dim,input_dim) for i in range(num_layers)])
        # Weighted Activations
        self.z = np.empty(num_layers, dtype=object)
        self.a = np.empty(num_layers, dtype=object)
        self.errors = np.asarray([np.zeros((input_dim,1)) for i in range(num_layers)])
    
    def compute(self,input,layer):
        weighted_input = np.dot(self.weights[layer],input) + self.bias[layer]
        return activation(weighted_input),weighted_input
    
    def feedforward(self,input_array,layer=0):
        # Terminate recursion once at the end
        if layer == num_layers:
            #self.z = np.asarray(self.z)
            #self.a = np.asarray(self.a)
            return input_array
        else:
            # Weighted activation output of layer l
            a_l, z_l = self.compute(input_array,layer)
            #self.z.append(z_l)
            #self.a.append(a_l)
            self.z[layer] = z_l
            self.a[layer] = a_l

            return self.feedforward(a_l,layer+1)
        
    def backpropogation(self,error_l,layer=num_layers-1):
        if layer == -1:
            #self.errors = np.asarray(self.errors)
            return
        else:
            error = (np.dot(self.weights[layer].T,error_l))*activation_derivative(self.z[layer-1])

            self.errors[layer] = error
            return self.backpropogation(error,layer-1)
    def updateParameters(self,input_array):
        # Update biases
        self.bias -= self.errors

        # Update weights
        for l in range(num_layers):
            for j in range(input_dim):
                for k in range(input_dim):
                    if l-1 == -1:
                        self.weights[l][j][k] -= learning_rate*input_array[k] * self.errors[l][j]
                    else:
                        self.weights[l][j][k] -= learning_rate*self.a[l-1][k] * self.errors[l][j]

    def cycle(self,input_array,true_output):
        # Get activation input, activation output
        output = self.feedforward(input_array)

        # Calculate output layer errors
        dL = (output-true_output) * activation_derivative(n.z[-1])

        # Backpropogation
        self.backpropogation(dL)
        
        # Update parameters
        self.updateParameters(input_array)

        #self.z = self.z.tolist()
        #self.a = self.a.tolist()

    def predict(self,input_array):
        return self.feedforward(input_array)


n = Network(num_layers,input_dim)

print("Input")
print(input_array)

num_epochs = 5

for epoch in range(num_epochs):
    # Prediction:
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("Cost: ", cost(n.predict(input_array)))
    print("Prediction: ", n.predict(input_array))
    n.cycle(input_array,true_output)


