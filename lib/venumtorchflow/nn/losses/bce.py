import numpy as np
from .loss import Loss


class Loss_BinaryCrossentropy(Loss):
    """ Compute the loss given the predictions and the labels. """

    def forward(self, y_pred, y_true):
        """"
        compute the loss given the predictions and the labels.

        parameters:
        -----------
        y_pred: np.array
            The predictions.
        y_true: np.array
            The labels.
        """
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Return the gradients of the loss with respect to the prediction. 

        parameters:
        ----------
        dvalues: np.array
            The gradients recieved for this layer
        y_true: np.array
            The labels.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
