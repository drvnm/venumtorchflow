import numpy as np
import nnfs

nnfs.init()


class Activation_Linear():
    def forward(self, inputs):
        # onthoud de waardens
        self.inputs = inputs
        self.output = inputs

    # derivative van linear functie
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
