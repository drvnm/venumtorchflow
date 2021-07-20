import numpy as np

class Optimizer_Adagrad:
    """
    initialize the optimizer
    """
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        """
        parameters
        ----------
        learning_rate: float, optional (default=1.)
            the learning rate for the optimizer
        decay: float, optional (default=0.)
            the rate at which the learning rate is decayed
        epsilon: float, optional, optional (default=1e-7)
            the small value added to the denominator to prevent division by zero
        """
        self.learning_rate = learning_rate
        # de learning rate die we uptdaten
        self.current_learning_rate = learning_rate
        # de hoeveelheid decay
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

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

        # als de layer geen gecachde arrays heeft, maak deze
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update de cache met squared huidige gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # update parameters
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Updates the iteration counter.
        call this after updating the parameters.
        """
        self.iterations += 1
