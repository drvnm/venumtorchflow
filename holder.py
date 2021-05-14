import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # maakt de weigts, inputs x neuronen zodat we de matrix niet hoeven te transposen.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # een column matrix, begin met 0
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # bereken de nieuwe outputs
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLu:
    def forward(self, inputs):
        # berekent ReLu op elk punt in de inputs
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # exponentiate elke waarde in de batch (e^x)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # voor elke sample in de batch, devide elke feature met de sum van alle features (gexponentiate)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # gemiddelde loss
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # zorgt ervoor dat je geen kleine waarden dan 0 en 1 krijgt
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # krijg alle hoogste confidence scores
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # krijg alle hoogste confidence scores
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # neem de natural log van elk element in de confidence scores
        neg_losses = -np.log(correct_confidences)
        return neg_losses


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
