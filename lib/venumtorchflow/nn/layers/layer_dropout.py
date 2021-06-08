import numpy as np
import nnfs

nnfs.init()


class Layer_Dropout:
    def __init__(self, rate):
        # voor 0.1 dropout hebben we 0.9 success
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        # maakt de binary_mask, zodat we die later verm. met de inputs
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
