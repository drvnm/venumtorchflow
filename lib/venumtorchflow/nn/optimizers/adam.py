import numpy as np


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        """ 
        Initialize the optimizer.

        parameters
        ----------
        learning_rate: float, optional (default=0.001)
            The learning rate.
        decay: float, optional (default=0.)
            The decay of the learning rate.
        epsilon: float, optional (default=1e-7)
            The epsilon value used for numerical stability.
        beta_1: float, optional (default=0.9)
            The beta_1 value used for the momentums.
        beta_2: float, optional (default=0.999)
            The beta_2 value used for the momentums

        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

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
        if not hasattr(layer, 'weight_cache'):
            # zet alle momentums en caches
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # zet de momentums
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + (1 - self.beta_1) * layer.dweights

        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # correct de momentums
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        # update cache met squared gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights ** 2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases ** 2

        # correct de caches
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """
        Updates the iteration counter.
        call this after updating the parameters.
        """
        self.iterations += 1
