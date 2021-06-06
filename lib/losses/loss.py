import numpy as np
import nnfs

nnfs.init()


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # gemiddelde loss
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        # 0 default
        regularization_loss = 0

        # L1 en L2 Regularization weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

        # L1 en L2 Regularization biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * layer.biases)

        return regularization_loss
