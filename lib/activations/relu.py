import numpy as np
import nnfs

nnfs.init()


class Activation_ReLu:
    def forward(self, inputs):
        # onthoud de waarden op de derivatives te berekenen
        self.inputs = inputs
        # berekent ReLu op elk punt in de inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # self.dinputs zij de derivatives w.r.t de relu functie * gradient
        self.dinputs[self.inputs <= 0] = 0
