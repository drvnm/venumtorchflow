import numpy as np
from .loss import Loss
import nnfs

nnfs.init()


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        # bereken loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(y_true)
        outputs = len(dvalues[0])

        # gradients
        self.dinputs = -2 * (y_true - dvalues) / outputs
        
        self.dinputs = self.dinputs / samples
