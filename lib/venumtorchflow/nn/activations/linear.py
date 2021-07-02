import numpy as np


class Activation_Linear():
    def forward(self, inputs, training):
        # onthoud de waardens
        self.inputs = inputs
        self.output = inputs

    # derivative van linear functie
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs
