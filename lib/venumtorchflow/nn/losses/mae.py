import numpy as np
from .loss import Loss


class Loss_MeanAbsoluteError(Loss):
    """
    calculates the loss of a network
    """

    def forward(self, y_pred, y_true):
        """
        calculates the loss of a network outputted by a network

        Parameters
        ----------
        y_pred : np.array
            network outputted values
        y_true : np.array
            desired output values
        """
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        calculates the gradients of the loss with respect to the network
        outputted values

        Parameters
        ----------
        dvalues : np.array
            gradient of the loss with respect to the network outputted values
        y_true : np.array
            desired output values
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
