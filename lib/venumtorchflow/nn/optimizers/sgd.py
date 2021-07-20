import numpy as np


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        """"
        Initializes the SGD optimizer

        parameters
        ----------
        learning_rate: float
            The learning rate for the optimizer.
        decay: float
            The decay rate for the learning rate, 
            indicates how fast the learning rate should decay.

        momentum: float
            The momentum value.
            Indicates how much the past gradients should be taken into account
        """

        self.learning_rate = learning_rate
        # de learning rate die we uptdaten
        self.current_learning_rate = learning_rate
        # de hoeveelheid decay
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """
        Updates the current learning rate if decay is used.
        call this before updating the parameters.
        """
        if self.decay:
            # update de learning rate, wordt steeds lager
            if self.decay:
                self.current_learning_rate = self.learning_rate * \
                    (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """	
        Updates the layer parameters.

        parameters:
        -----------
        layer: nn.layer()
            The layer to update, must have weights or biases.
        """

        # als we SGD met momentum gebruiken
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # update de weights en biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Updates the iteration counter.
        call this after updating the parameters.
        """
        self.iterations += 1
