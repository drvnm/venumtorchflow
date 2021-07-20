import numpy as np
from .accuracy import Accuracy


class Accuracy_Categorical(Accuracy):
    """
    accuracy to use for categoical data
    """

    def __init__(self, *, binary=False):
        """
        initalizes accuracy object

        parameters
        ----------
        binary: bool
            whether or not the data is binary
        """
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        """
        compares the predictions to the actual outputs

        parameters:
        -----------
        predictions: np.array
            predictions outputted by the network
        y: np.array
            actual outputs for the data
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
