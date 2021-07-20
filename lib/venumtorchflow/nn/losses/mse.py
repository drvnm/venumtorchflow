import numpy as np
from .loss import Loss


class Loss_MeanSquaredError(Loss):
    """
    caluculates the loss of the network
    """

    def forward(self, y_pred, y_true):
        """
        Calculates the loss of the network outputted values 
        compared to the desired output y_true

        parameters:
        -----------

        y_pred: np.array
            The network outputted values
        y_true: np.array
            The desired output values
        """
        # bereken loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=1)
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
        samples = len(y_true)
        outputs = len(dvalues[0])

        # gradients
        self.dinputs = -2 * (y_true - dvalues) / outputs

        self.dinputs = self.dinputs / samples
