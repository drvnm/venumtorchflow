import numpy as np


class Activation_ReLu:
    """
    ReLu activation function
    """

    def forward(self, inputs, training):
        """
        forward propagation function for ReLu activation function

        parameters:
        ----------
        inputs: np.ndarray
            inputs
        training: bool
            whether the network is in training mode or not
        """
        # onthoud de waarden op de derivatives te berekenen
        self.inputs = inputs
        # berekent ReLu op elk punt in de inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        backward pass for ReLu activation function

        parameters
        ----------
        dvalues: np.array
            derivatives from the next layer
        """
        self.dinputs = dvalues.copy()
        # self.dinputs zij de derivatives w.r.t de relu functie * gradient
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs
