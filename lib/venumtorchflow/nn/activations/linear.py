import numpy as np


class Activation_Linear():
    """
    linear activation
    """

    def forward(self, inputs, training):
        """
        forward pass

        parameters
        ----------
        inputs: np.array
            input values for this layer
        training: bool
            whether or not we are in training mode
        """
        # onthoud de waardens
        self.inputs = inputs
        self.output = inputs

    # derivative van linear functie
    def backward(self, dvalues):
        """
        backward pass for linear activation function

        parameters
        ---------
        dvalues: np.array
            derivatives from the next layer
        """
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs
