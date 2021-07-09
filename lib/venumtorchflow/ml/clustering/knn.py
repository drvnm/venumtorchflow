import pandas as pd
import numpy as np
from scipy.stats import mode

# TODO: make this go brr 
class KNearestNeighbors:
    def __init__(self, k, training_x, training_y):
        # k is het aantal punten dat we
        # van het nieuwe data punt berekenen (afstand)
        self.k = k
        self.x = training_x
        self.y = training_y
        self.sample_amount, self.features = self.x.shape

    def predict(self, x):
        self.predicts = x
        print(x.shape)
        if len(x.shape) == 1:
            self.psample_amount = 1
        else:
            self.psample_amount, self.pfeatures = x.shape

        # pfirst is het aantal samples, dus pvalues
        # wordt het aantal values die we moeten predicten
        predicted_values = np.zeros(self.psample_amount)

        for i in range(self.psample_amount):
            # de Ith sample in de gegeven data
            current_sample = self.predicts[i]
            print(current_sample)
            # vind de k nearest classifications die bij het punt horen
            neigbors = self.find_neighbors(current_sample)
            predicted_values[i] = mode(neigbors)[0][0]
        return predicted_values

    def find_neighbors(self, sample):
        # lege array voor elk punt in onze training data
        distances = np.zeros(self.sample_amount)
        for i in range(self.sample_amount):
            distance = np.linalg.norm(sample - self.x[i])
            # Ith distance van sample naar training data
            distances[i] = distance

        inds = distances.argsort()
        y_sorted = self.y[inds]
        return y_sorted[:self.k]

