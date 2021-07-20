
import pickle
import copy
import numpy as np

import sys
sys.path.append("../layers") 
sys.path.append("../activations") 
sys.path.append("../losses") 


class Model:
    def __init__(self):
        """
        initialize model
        """
        # slaat alle layers op (dense, activation)
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        """
        add a layer to the model

        parameters
        ----------
        layer: layer object
            the layer to add
        """
        self.layers.append(layer)

    # zet de loss en optimizer
    def set(self, *, loss, optimizer, accuracy):
        """
        set loss, optimizer and accuracy for the model

        parameters
        ----------
        loss: loss object
            the loss to use for the model
        optimizer: optimizer object
            the optimizer to use for the model
        accuracy: accuracy object
            the accuracy to use for the model
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1,
              validation_data=None):
        """
        train the model for a given number of epochs, with a given batch size

        parameters
        ----------
        X: numpy array
            the training data
        y: numpy array
            the training labels, one hot encoded or sparse
        epochs: int
            the number of epochs to train for
        batch_size: int
            the size of the batches for training
        print_every: int
            print the loss and accuracy every `print_every` epochs
        validation_data: tuple (<numpy array>, <numpy array>)
            if provided, the model will be evaluated against this data every `print_every` epochs
        """

        self.batch_size = batch_size
        # zet accuracy parameters
        self.accuracy.init(y)

        # default train step als er geen batch is (alle data in een keer)
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # epoch loop
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            # reset sum van loss en accuracy
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # als er weel batches zijn
                else:
                    batch_X = X[step * batch_size: (step + 1) * batch_size]
                    batch_y = y[step * batch_size: (step + 1) * batch_size]

                output = self.forward(batch_X, training=True)

                # losses
                data_loss, reg_loss = \
                    self.loss.calculate(
                        output, batch_y, include_regularization=True)
                loss = data_loss + reg_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # back prop
                self.backward(output, batch_y)

                # optimize met aangegeven optimizer
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'    step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {reg_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # epoch info
            epoch_data_loss, epoch_reg_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_acc:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_reg_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def finalize(self):
        """
        this method will set all of the needed properties, call before training
        """
        # eigen input layer
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            # als i 0 is zijn we in de eerste hidden layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # alle layers behalven 1e en laatste
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # laatste layer, next is loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # check of we de layer kunnen trainen
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and \
                isinstance(self.loss, Loss_CategoricalCrossentropy):
            # als laatste layer softmax en CCE is, maak comnbined activation
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossEntropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            # back prop voor combined activation
            self.softmax_classifier_output.backward(output, y)

            # omdat we niet backward callen op de laatste layer zetten we de dinputs
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # ga van output layer naar input layer, krijg elke gradient
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        # gradients voor loss functie
        self.loss.backward(output, y)

        # ga van output layer naar input layer, krijg elke gradient
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[
                    step * batch_size: (step + 1) * batch_size
                ]

                batch_y = y_val[
                    step * batch_size: (step + 1) * batch_size
                ]

            output = self.forward(batch_X, False)

            # loss en acc voor validatie data
            loss = self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(
                output)
            accuracy = self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def get_parameters(self):
        # krijg alle weights en biases van elke trainable layers. [(weights, biases)..]
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            # * parameter_set wordt (weights, biases)
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        """
        save parameters to pickle file

        parameters
        ----------
        path: str
            path to save parameters to
        """
        # write alle parameters naar pickle file
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        """
        save model to file

        parameters
        ----------
        path: str
            path to save model to
        """
        # reset alles voor de model
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        possible_attr = ['inputs', 'output', 'dinputs',
                         'dweights', 'dbiases']

        for layer in model.layers:
            for _property in possible_attr:
                layer.__dict__.pop(_property, None)

        # sla de hele model op
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        """ loads model from file """
        # lees de binary voor de model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model  # laad een hele model in

    def predict(self, X, *, batch_size=None):
        """ 
        Predict the output for the given input.

        parameters
        ----------
        X: numpy array
            The input to the network
        batch_size: int
            The batch size to use for the prediction, if None use the whole dataset will be predicted
        """
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            # als er geen batch size is is X de hele batch
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size: (step + 1) * batch_size]
        batch_output = self.forward(batch_X, training=False)
        output.append(batch_output)

        # [[1, 2], [3, 4]], [[5, 6], [7,8]] = [[1,2], [3, 4], [5, 6].. etc]
        return np.vstack(output)

    def __repr__(self):
        # maakt een summary van een model object
        total_tunable = 0
        print("------------------------------------------------------------")
        print("     Layer type          Output size         Tunable params")
        print("============================================================")
        for layer in self.trainable_layers:
            layer_output_size = str((self.batch_size or 1,
                                     layer.num_output_neurons))
            print(
                "     Layer_Dense,       {:10s}           {:10s}".format(layer_output_size, str(layer.tunable_params)))
            total_tunable += layer.tunable_params

        print("============================================================")
        print(f"total tunable params: {total_tunable}")
        return ''
