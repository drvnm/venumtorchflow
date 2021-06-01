
# # import numpy as np

# # dvalues = np.array([[1., 1., 1.],
# #                     [2., 2., 2.],
# #                     [3., 3., 3.]])


# # weights = np.array([[0.2, 0.8, -0.5, 1],
# #                     [0.5, -0.91, 0.26, -0.5],
# #                     [-0.26, -0.27, 0.17, 0.87]]).T

# # dinputs = np.dot(dvalues, weights.T)

# # # ----------------------------------------------------------------
# # inputs = np.array([[1, 2, 3, 2.5],
# #                    [2., 5., -1., 2],
# #                    [-1.5, 2.7, 3.3, -0.8]])

# # # sum inputs for given weight
# # # and multiply by the passed-in gradient for this neuron
# # dweights = np.dot(inputs.T, dvalues)


# # # ----------------------------------------------------------------
# # biases = np.array([[2, 3, 0.5]])
# # dbiases = np.sum(dvalues, axis=0, keepdims=True)

# # # ----------------------------------------------------------------
# # # Example layer output
# # z = np.array([[1, 2, -3, -4],
# #               [2, -7, -1, 3],
# #               [-1, 2, 5, -1]])

# # dvalues = np.array([[1, 2, 3, 4],
# #                     [5, 6, 7, 8],
# #                     [9, 10, 11, 12]])

# # drelu = dvalues.copy()
# # drelu[z <= 0] = 0
# # # print(drelu)

# # layer_output = np.dot(inputs, weights) + biases
# # relu_output = np.maximum(0, layer_output)
# # drelu = relu_output.copy()
# # drelu[layer_output <= 0] = 0

# # dipunts = np.dot(drelu, weights.T)
# # dweights = np.dot(inputs.T, drelu)
# # dbiases = np.sum(drelu, axis=0, keepdims=True)


# # weights += -0.001 * dweights
# # biases += -0.001 * dbiases

# # print(layer_output)

# import numpy as np


# softmax_output = [0.7, 0.1, 0.2]
# softmax_output = np.array(softmax_output).reshape(-1, 1)
# print(softmax_output)
# print(np.diagflat(softmax_output) -
# np.dot(softmax_output, softmax_output.T))


starting_learning_rate = 1
learning_rate_decay = 0.1
step = 20

learning_rate = starting_learning_rate * \
    (1. / (1 + learning_rate_decay * step))
print(learning_rate)
