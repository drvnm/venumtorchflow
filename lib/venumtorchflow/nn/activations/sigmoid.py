import numpy as np


class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        # sigmoid op inputs
        self.output = 1 / (1 + np.exp(-self.inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
