import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt


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
        # onthoud de inputs
        self.inputs = inputs

    def backward(self, dvalues):
        # gradients voor de weights en biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradients met w.r.t inputs
        self.dipunts = np.dot(dvalues, self.weights.T)


class Activation_ReLu:
    def forward(self, inputs):
        # onthoud de waarden op de derivatives te berekenen
        self.inputs = inputs
        # berekent ReLu op elk punt in de inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.diputs = dvalues.copy()
        # self.diputs zij de derivatives w.r.t de relu functie * gradient
        self.diputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        # exponentiate elke waarde in de batch (e^x)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # voor elke sample in de batch, devide elke feature met de sum van alle features (gexponentiate)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


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

        # wanneer de y_true alleen de indexen van de correcte waardens bevat, en niet de hele array.
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
        self.diputs = -y_true / dvalues
        # normalize gradient
        self.diputs = self.diputs / samples


# ============================ chapter 6 optimize ==
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# # loss functio22n
loss_function = Loss_CategoricalCrossentropy()

# helper variables
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Generate een nieuwe set of weights en biases voor elke iteration
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # bereken loss
    loss = loss_function.calculate(activation2.output, y)

    # bereken accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    if loss < lowest_loss:
        print('New set of weights found, iteration: ', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense2_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
