Shallow Neural Network Notes

⸻

1. initialize_parameters

Q1. Why do W1, W2 use np.random, but b1, b2 use np.zeros?

Setting initial bias as 0 is okay because gradient descent updates it.

However, if I set weights as 0, all nodes of the hidden layer do the same calculation.

Hence, backward propagation gets the same gradient -> Symmetry Breaking


⸻

Q2. Why multiply by np.sqrt(2. / n_x)?

If W is randomly chosen, its value may be too large.

For that, Z1 and Z2 may blow up, so sigmoid and tanh do not work well.

Therefore, np.sqrt(2. / n_x) is multiplied.

This is called 'He Initialization'

It scales the weights properly.

⸻

Q3. What does parameters do?

parameters is used to group variables into a Python dictionary.

For example:

parameters = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
}

Later, we can call each variable by writing:

W1 = parameters["W1"]
b1 = parameters["b1"]

and so on.

⸻

2. forward_propagation

Q1. Input for the function?

The inputs are:

X, parameters

X is the input data.

Since it is a one-hidden-layer neural network, the input can be understood as the previous layer output.

parameters is used to call weights and bias.

⸻

Q2. Activation function?

From input layer to hidden layer:

tanh function

This is a non-linear activation function.

From hidden layer to output layer:

sigmoid function

⸻

Q3. Cache?

cache is similar to the parameters dictionary.

It saves:

Z1, A1, Z2, A2

for back propagation.

Recall that backward propagation uses:

dZ2 = A2 - Y
dW2 = (1/m) * np.dot(dZ2, A1.T)
dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))

Therefore, A1 and A2 should be saved during forward propagation.

⸻

3. compute_cost

The purpose of this function is to measure how different Y, the actual answer, and Y_hat, the estimated answer, are.

For example:

A2 = [0.91, 0.12, 0.78, ...]
Y  = [1,    0,    1,    ...]

⸻

Q1. Binary Cross-Entropy Cost Function

Suppose the actual answer is:

Y = 1

and the prediction is close to 1.

Then the cost function becomes:

-log(0.99)

which is really small.

However, if the prediction is close to 0, then the cost function becomes:

-log(0.01)

which is really big.

Hence, if the prediction is wrong, for both 0 and 1, the cost function gets large.

⸻

4. backward_propagation

WTS:

dW1, db1, dW2, db2

The formulas are:

dZ2 = A2 - Y
dW2 = (1/m) * np.dot(dZ2, A1.T)
db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
dW1 = (1/m) * np.dot(dZ1, X.T)
db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

For mathematical intuition, check GoodNotes.

⸻

Q2. Grads

grads is a dictionary for:

dW1, db1, dW2, db2

It is used in update_parameters.

For example:

grads = {
    "dW1": dW1,
    "dW2": dW2,
    "db1": db1,
    "db2": db2
}