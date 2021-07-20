import numpy as np


class Activation_Sigmoid:
    """
    sigmoid activation
    """

    def forward(self, inputs, training):
        """
        forward propagation with sigmoid function

        parameters
        ----------
        inputs: np.array
            inputs
        training: bool
            whether to use training mode or not
        """
        self.inputs = inputs
        # sigmoid op inputs
        self.output = 1 / (1 + np.exp(-self.inputs))

    def backward(self, dvalues):
        """
        backward pass for sigmoid activation function

        parameters
        ----------
        dvalues: np.array
            derivatives from the next layer
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
