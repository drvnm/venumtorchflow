from .accuracy import Accuracy
import numpy as np


class Accuracy_Regression(Accuracy):
    """
    Accuracy to use if network is a regrssion model
    """

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        """
        Initialize the precision of the accuracy.

        parameters
        ----------
        y: np.array
            Target values.
        reinit: bool
            If true, the precision will be reinitialized.
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Compare the predicted values to the target values.

        parameters:
        -----------
        predictions: np.array
            Predicted values.
        y: np.array
            Target values.
        """
        return np.absolute(predictions - y) < self.precision
