import numpy as np

np.random.seed(42)

# Width of the input layer
input_dimensions = 2

# Dimensions of the hidden layers
hidden_layers = [1]

# Width of the output layer
output_dimensions = 1

# Overall structure
structure = [input_dimensions] + hidden_layers + [output_dimensions]

# Generate weights and biases

# Randomly initiate weights
weights = [np.random.rand(structure[l],structure[l+1]) for l in range(len(structure[1:-1]))]

# Randomly generate biases
biases = [np.ones(l) for l in structure[1:-1]]

print(biases)