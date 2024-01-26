import numpy as np

np.random.seed(42)

# Number of neurons in each layer
layer_width = [1,2]
# Layer 1 is the input (indexed as 0)
# Layer 2 is the hidden layers 
# Layer 3 is the output

class Network:
    def __init__(self,layer_width):
        # Biases matrix
        # Every neuron has a bias
        self.bias = np.asarray([np.ones(i) for i in layer_width],dtype=object)

        # Weights matrix
        self.weights = np.asarray([np.random.rand(layer_width[i],layer_width[i]) if i==0 else np.random.randn(layer_width[i],layer_width[i-1]) for i in range(len(layer_width))],dtype=object)


    def activation(self,x):
        return 1/(1+np.exp(-x))
    
    def activation_prime(self,x):
        return self.activation(x)*(1-self.activation(x))
    
    def cost(self,y,y0):
        return 0.5 * (y0-y)**2

    def layer_compute(self,layer_input,l):
        weighted_input = np.dot(self.weights[l],layer_input)+self.bias[l]
        weighted_activation = self.activation(weighted_input)
        return weighted_activation,weighted_input
    
    def feedforward(self,layer_input):
        self.a = []
        self.z = []
        for l in range(len(layer_width)):
            # Skip first layer (no layer before the first, so there are no weights)
            #if l == 0:
            #    self.a.append(np.ones(len(layer_input)))
            #    self.z.append(np.ones(len(layer_input)))
            #    continue
            
            # Compute the layer
            a_l, z_l = self.layer_compute(layer_input,l)
            
            # Save the weighted activations and weighted inputs for use in backprop
            self.a.append(a_l)
            self.z.append(z_l)

            # Update the input to the layer
            layer_input = a_l
        return self.a[-1]
    
    def delta_l(self,l,delta_lp1):
        return np.dot(self.weights[l+1].T,delta_lp1) * self.activation_prime(self.z[l])

    def backpropagation(self,true_output):
        delta_L = (self.a[-1]-true_output) * self.activation_prime(self.z[-1])
        
        delta = []
        delta.insert(0,delta_L)
        for l in range(len(layer_width)-2,-1,-1):
            delta_l = self.delta_l(l,delta_L)
            delta.insert(0,delta_l)
            delta_L = delta_l
        
        return delta
    
    def updateParams(self,delta,lr):
        # Update biases layer by layer
        for l in range(len(layer_width)):
            self.bias[l] = self.bias[l]  - lr * delta[l]

        # Update weights
        for l in range(1,len(layer_width)):
            for j in range(layer_width[l]):
                for k in range(layer_width[l-1]):
                    self.weights[l][j][k] = self.weights[l][j][k] - self.a[l-1][k] * lr *delta[l][j]

    def cycle(self,input,true_output,lr):
        output = self.feedforward(input)
        delta = self.backpropagation(true_output)
        print("Delta")
        print(delta)
        self.updateParams(delta,lr)
        return self.cost(input,true_output)


n = Network(layer_width)

network_input = np.asarray([1])
true_output = np.asarray([0])

learning_rate = 1e200
epochs = 2

out = n.feedforward(network_input)
print(n.bias)
print("")
print(n.weights)
print("")
print(out)


quit()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    cost = n.cycle(network_input,true_output,learning_rate)
    print(f"Cost: {cost}")
    print(n.weights[0])
    print()

quit()

print("Weights before")
print(n.weights)

print("Bias before")
print(n.bias)

print("Output")
print(n.feedforward(network_input))
delta = n.backpropagation(true_output)
n.updateParams(delta)

print("Weights after")
print(n.weights)

print("Bias after")
print(n.bias)
