import numpy as np
from .loss import Loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # zorgt ervoor dat je geen kleine waarden dan 0 en 1 krijgt
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # wanneer de y_true alleen de indexen van de correcte waardens bevat,
        # en niet de hele array.
        if len(y_true.shape) == 1:
            # krijg alle hoogste confidence scores
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # krijg alle hoogste confidence scores
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # neem de natural log van elk element in de confidence scores
        neg_losses = -np.log(correct_confidences)
        return neg_losses

    def backward(self, dvalues, y_true):
        # hvlheid samples
        samples = len(dvalues)
        labels = len(dvalues[0])

        # maak y_true one hot encoded als dit niet zo is.
        if(len(y_true.shape)) == 1:
            y_true = np.eye(labels)[y_true]

        # bereken de gradient
        self.dinputs = -y_true / dvalues
        # normalize gradient
        self.dinputs = self.dinputs / samples
