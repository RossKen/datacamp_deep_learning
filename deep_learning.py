#########################################################
# Exercise 1 - Coding Forward Propogation - Single Layer 

# Essentially just setting up a dot product of the inputs and weights for each object in the hidden layer, then summing the result

import numpy as np

# define input data and weights
input_data = np.array([3, 5])
type(input_data)

weights = {'node_0': np.array([2, 4]), 'node_1': np.array([4, -5]), 'output': np.array([2, 7])}

type(weights)
type(weights['node_0'])

a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.dot(a, b)

# Calculate node values

node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs

hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output

output = (hidden_layer_outputs * weights['output']).sum()

print(output)


#########################################################
# Exercise 2 -  Coding Forward Propogation with an activation function

# In reality, most inputs have do not have a simple, linear contribution to our output (as used in example 1). Now we use an activation function to determine how the inputs of our hidden layer (i.e. the raw values multiplied by their weights within the networks) are transformed. In this example we are going to use a Recified Linear activation function. Sigmoid functions are also popular, e.g. tanh

# What is a Rectified Linear Activation Function (RELU)?
# y = 0 for x > 0
# y = x for x >= 0

#
#                 /
#                /
#               /
#              /
#   __________/
#            0

# Work this alteration into the solution for exercise 1

# define RELU function

def relu(input):
    if input < 0:
        output = 0
    else:
        output = input
        
    return(output)

# OR

def relu(input):
    output = max(input, 0)
    return(output)
    

# calculate node values

node_0_input = (input_data * weights["node_0"]).sum()
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights["node_0"]).sum()
node_1_output = relu(node_1_input)

# wrap prediction code into single function

def predict_with_network(input_data_row, weights):
    
    #node 0
    node_0_input = (input_data_row * weights["node_0"]).sum()
    node_0_output = relu(node_0_input)

    #node 1
    node_1_input = (input_data_row * weights["node_1"]).sum()
    node_1_output = relu(node_1_input)
    
    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights["output"]). sum()
    model_output = relu(input_to_final_layer)
                            
    # Return model output
    return(model_output)

# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)


#########################################################
# Exercise 3 - Coding Forward Propogation - Multiple Layers

