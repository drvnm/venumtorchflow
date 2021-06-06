import numpy as np
from .softmax import Activation_Softmax
from losses.cce import Loss_CategoricalCrossentropy
import nnfs

nnfs.init()


class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # forward naar softmax
        self.activation.forward(inputs)
        self.output = self.activation.output
        # return de loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues,  y_true):
        # # van samples
        samples = len(dvalues)

        # maak one-hot encoded discrete waardens.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
