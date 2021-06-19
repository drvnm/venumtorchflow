import numpy as np
import nnfs

nnfs.init()


class Activation_Sigmoid:
    def __init__(self, inputs):
        self.inputs = inputs
        # sigmoid op inputs
        self.output = 1 / (1 + np.exp(-self.inputs))

    def backward(self, dvalues):
        self.dipunts = dvalues * (1 - self.output) * self.output
