import numpy as np


class Layer_Dropout:
    """
    a dropout layer
    """

    def __init__(self, rate):
        """
        initializes the layer with a given dropout rate (0.0 <= rate <= 1.0)

        parameters
        ----------
        rate: float
            dropout rate
        """
        # voor 0.1 dropout hebben we 0.9 success
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """
        forward propagation with dropout

        parameters
        ----------
        inputs: np.array
            inputs to layer
        training: bool
            whether or not we are in training mode
        """
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        # maakt de binary_mask, zodat we die later verm. met de inputs
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """
        backward propagation

        parameters
        ----------
        dvalues: np.array
            derivatives from upper layer
        """
        self.dinputs = dvalues * self.binary_mask
