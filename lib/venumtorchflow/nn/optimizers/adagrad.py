import numpy as np

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        # de learning rate die we uptdaten
        self.current_learning_rate = learning_rate
        # de hoeveelheid decay
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            # update de learning rate, wordt steeds lager
            if self.decay:
                self.current_learning_rate = self.learning_rate * \
                    (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # als de layer geen gecachde arrays heeft, maak deze
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update de cache met squared huidige gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        print(layer.weight_cache)

        # update parameters
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
