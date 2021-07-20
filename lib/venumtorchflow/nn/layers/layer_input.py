class Layer_Input:
    """
    Input layer to any network
    """

    def forward(self, inputs, training):
        """
        Forward pass for input layer

        parameters
        ----------
        inputs: ndarray
            Inputs to the layer.
        training: bool
            whether training is active or not.
        """
        self.output = inputs
