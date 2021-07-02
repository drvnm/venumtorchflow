import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # maakt de weigts, inputs x neuronen zodat we de matrix
        # niet hoeven te transposen.
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # een column matrix, begin met 0
        self.biases = np.zeros((1, n_neurons))

        # zet alle regularization parameters
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        # slaat op hoeveel weigts en biases we kunnen veranderen
        self.tunable_params = n_inputs * n_neurons + len(self.biases)
        self.num_output_neurons = n_neurons

    def forward(self, inputs, training):
        # onthoud de inputs
        self.inputs = inputs
        # bereken de nieuwe outputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # gradients voor de weights en biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 op weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 op weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.weights

        # L1 op biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 op biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases

        # gradients met w.r.t inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        # geeft laatste weights en biases, later saven
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
