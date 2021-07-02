import numpy as np


class Activation_Softmax:
    def forward(self, inputs, training):
        # exponentiate elke waarde in de batch (e^x)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # voor elke sample in de batch, devide elke feature met de sum van alle features (gexponentiate)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # maak lege array
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            # krijg de jacobian matrix
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
