import numpy as np


class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        """
        Initialize the optimizer.

        parameters
        ----------
        learning_rate : float, optional (default=0.001)
            The learning rate.
        decay : float, optional (default=0.)
            The decay rate for the learning rate, 
            indicates how fast the learning rate should decay.
        epsilon : float, optional (default=1e-7)
            A small constant used for numerical stability.
        rho : float, optional (default=0.9)
            The momentum factor.

        """

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        """
        Updates the current learning rate if decay is used.
        call this before updating the parameters.
        """

        # als we decay hebben, maak de learning rate dan steeds iets kleiner
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

        # als de layer nog geen cached gradients heeft, maak deze dan
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update caches
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights ** 2

        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases ** 2

        # update weights en biases met SGD
        layer.weights += -self.current_learning_rate * \
            layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * \
            layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Updates the iteration counter.
        call this after updating the parameters.
        """
        self.iterations += 1
