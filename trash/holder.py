# import numpy as np
# import nnfs
# from nnfs.datasets import spiral_data


# nnfs.init()


# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons,
#                  weight_regularizer_l1=0, weight_regularizer_l2=0,
#                  bias_regularizer_l1=0, bias_regularizer_l2=0):
#         # maakt de weigts, inputs x neuronen zodat we de matrix niet hoeven te transposen.
#         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
#         # een column matrix, begin met 0
#         self.biases = np.zeros((1, n_neurons))

#         # zet alle regularization parameters
#         self.weight_regularizer_l1 = weight_regularizer_l1
#         self.weight_regularizer_l2 = weight_regularizer_l2
#         self.bias_regularizer_l1 = bias_regularizer_l1
#         self.bias_regularizer_l2 = bias_regularizer_l2

#     def forward(self, inputs):
#         # onthoud de inputs
#         self.inputs = inputs
#         # bereken de nieuwe outputs
#         self.output = np.dot(inputs, self.weights) + self.biases

#     def backward(self, dvalues):
#         # gradients voor de weights en biases
#         self.dweights = np.dot(self.inputs.T, dvalues)
#         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

#         # L1 op weights
#         if self.weight_regularizer_l1 > 0:
#             dL1 = np.ones_like(self.weights)
#             dL1[self.weights < 0] = -1
#             self.dweights += self.weight_regularizer_l1 * dL1

#         # L2 op weights
#         if self.weight_regularizer_l2 > 0:
#             self.dweights += 2 * self.weight_regularizer_l2 * \
#                 self.weights

#         # L1 op biases
#         if self.bias_regularizer_l1 > 0:
#             dL1 = np.ones_like(self.biases)
#             dL1[self.biases < 0] = -1
#             self.dbiases += self.bias_regularizer_l1 * dL1

#         # L2 op biases
#         if self.bias_regularizer_l2 > 0:
#             self.dbiases += 2 * self.bias_regularizer_l2 * \
#                 self.biases

#         # gradients met w.r.t inputs
#         self.dinputs = np.dot(dvalues, self.weights.T)


# class Activation_ReLu:
#     def forward(self, inputs):
#         # onthoud de waarden op de derivatives te berekenen
#         self.inputs = inputs
#         # berekent ReLu op elk punt in de inputs
#         self.output = np.maximum(0, inputs)

#     def backward(self, dvalues):
#         self.dinputs = dvalues.copy()
#         # self.dinputs zij de derivatives w.r.t de relu functie * gradient
#         self.dinputs[self.inputs <= 0] = 0


# class Activation_Softmax:
#     def forward(self, inputs):
#         # exponentiate elke waarde in de batch (e^x)
#         exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         # voor elke sample in de batch, devide elke feature met de sum van alle features (gexponentiate)
#         probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#         self.output = probabilities

#     def backward(self, dvalues):
#         # maak lege array
#         self.dinputs = np.empty_like(dvalues)

#         for index, (single_output, single_dvalues) in \
#                 enumerate(zip(self.output, dvalues)):
#             single_output = single_output.reshape(-1, 1)
#             # krijg de jacobian matrix
#             jacobian_matrix = np.diagflat(
#                 single_output) - np.dot(single_output, single_output.T)
#             self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)




# class Loss_CategoricalCrossentropy(Loss):
#     def forward(self, y_pred, y_true):
#         samples = len(y_pred)

#         # zorgt ervoor dat je geen kleine waarden dan 0 en 1 krijgt
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

#         # wanneer de y_true alleen de indexen van de correcte waardens bevat, en niet de hele array.
#         if len(y_true.shape) == 1:
#             # krijg alle hoogste confidence scores
#             correct_confidences = y_pred_clipped[range(samples), y_true]

#         elif len(y_true.shape) == 2:
#             # krijg alle hoogste confidence scores
#             correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
#         # neem de natural log van elk element in de confidence scores
#         neg_losses = -np.log(correct_confidences)
#         return neg_losses

#     def backward(self, dvalues, y_true):
#         # hvlheid samples
#         samples = len(dvalues)
#         labels = len(dvalues[0])

#         # maak y_true one hot encoded als dit niet zo is.
#         if(len(y_true.shape)) == 1:
#             y_true = np.eye(labels)[y_true]

#         # bereken de gradient
#         self.dinputs = -y_true / dvalues
#         # normalize gradient
#         self.dinputs = self.dinputs / samples


# class Activation_Softmax_Loss_CategoricalCrossEntropy():
#     def __init__(self):
#         self.activation = Activation_Softmax()
#         self.loss = Loss_CategoricalCrossentropy()

#     def forward(self, inputs, y_true):
#         # forward naar softmax
#         self.activation.forward(inputs)
#         self.output = self.activation.output
#         # return de loss
#         return self.loss.calculate(self.output, y_true)

#     def backward(self, dvalues,  y_true):
#         # # van samples
#         samples = len(dvalues)

#         # maak one-hot encoded discrete waardens.
#         if len(y_true.shape) == 2:
#             y_true = np.argmax(y_true, axis=1)

#         self.dinputs = dvalues.copy()
#         self.dinputs[range(samples), y_true] -= 1
#         self.dinputs = self.dinputs / samples


# class Optimizer_SGD:
#     def __init__(self, learning_rate=1., decay=0., momentum=0.):
#         self.learning_rate = learning_rate
#         # de learning rate die we uptdaten
#         self.current_learning_rate = learning_rate
#         # de hoeveelheid decay
#         self.decay = decay
#         self.iterations = 0
#         self.momentum = momentum

#     def pre_update_params(self):
#         if self.decay:
#             # update de learning rate, wordt steeds lager
#             if self.decay:
#                 self.current_learning_rate = self.learning_rate * \
#                     (1. / (1. + self.decay * self.iterations))

#     def update_params(self, layer):
#         # als we SGD met momentum gebruiken
#         if self.momentum:
#             if not hasattr(layer, 'weight_momentums'):
#                 layer.weight_momentums = np.zeros_like(layer.weights)
#                 layer.bias_momentums = np.zeros_like(layer.biases)

#             weight_updates = \
#                 self.momentum * layer.weight_momentums - \
#                 self.current_learning_rate * layer.dweights
#             layer.weight_momentums = weight_updates
#             # Build bias updates
#             bias_updates = \
#                 self.momentum * layer.bias_momentums - \
#                 self.current_learning_rate * layer.dbiases
#             layer.bias_momentums = bias_updates
#         else:
#             weight_updates = -self.current_learning_rate * layer.dweights
#             bias_updates = -self.current_learning_rate * layer.dbiases

#         # update de weights en biases
#         layer.weights += weight_updates
#         layer.biases += bias_updates

#     def post_update_params(self):
#         self.iterations += 1


# class Optimizer_Adagrad:
#     def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
#         self.learning_rate = learning_rate
#         # de learning rate die we uptdaten
#         self.current_learning_rate = learning_rate
#         # de hoeveelheid decay
#         self.decay = decay
#         self.iterations = 0
#         self.epsilon = epsilon

#     def pre_update_params(self):
#         if self.decay:
#             # update de learning rate, wordt steeds lager
#             if self.decay:
#                 self.current_learning_rate = self.learning_rate * \
#                     (1. / (1. + self.decay * self.iterations))

#     def update_params(self, layer):

#         # als de layer geen gecachde arrays heeft, maak deze
#         if not hasattr(layer, 'weight_cache'):
#             layer.weight_cache = np.zeros_like(layer.weights)
#             layer.bias_cache = np.zeros_like(layer.biases)

#         # update de cache met squared huidige gradients
#         layer.weight_cache += layer.dweights ** 2
#         layer.bias_cache += layer.dbiases ** 2
#         print(layer.weight_cache)

#         # update parameters
#         layer.weights += -self.current_learning_rate * \
#             layer.dweights / \
#             (np.sqrt(layer.weight_cache) + self.epsilon)

#         layer.biases += -self.current_learning_rate * \
#             layer.dbiases / \
#             (np.sqrt(layer.bias_cache) + self.epsilon)

#     def post_update_params(self):
#         self.iterations += 1


# class Optimizer_RMSprop:
#     def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
#                  rho=0.9):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0
#         self.epsilon = epsilon
#         self.rho = rho

#     def pre_update_params(self):
#         # als we decay hebben, maak de learning rate dan steeds iets kleiner
#         if self.decay:
#             self.current_learning_rate = self.learning_rate * \
#                 (1. / (1. + self.decay * self.iterations))

#     def update_params(self, layer):
#         # als de layer nog geen cached gradients heeft, maak deze dan
#         if not hasattr(layer, 'weight_cache'):
#             layer.weight_cache = np.zeros_like(layer.weights)
#             layer.bias_cache = np.zeros_like(layer.biases)

#         # update caches
#         layer.weight_cache = self.rho * layer.weight_cache + \
#             (1 - self.rho) * layer.dweights ** 2

#         layer.bias_cache = self.rho * layer.bias_cache + \
#             (1 - self.rho) * layer.dbiases ** 2

#         # update weights en biases met SGD
#         layer.weights += -self.current_learning_rate * \
#             layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)

#         layer.biases += -self.current_learning_rate * \
#             layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

#     def post_update_params(self):
#         self.iterations += 1


# class Optimizer_Adam:
#     def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
#                  beta_1=0.9, beta_2=0.999):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0
#         self.epsilon = epsilon
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2

#     def pre_update_params(self):
#         # als we decay hebben, maak de learning rate dan steeds iets kleiner
#         if self.decay:
#             self.current_learning_rate = self.learning_rate * \
#                 (1. / (1. + self.decay * self.iterations))

#     def update_params(self, layer):
#         if not hasattr(layer, 'weight_cache'):
#             # zet alle momentums en caches
#             layer.weight_momentums = np.zeros_like(layer.weights)
#             layer.weight_cache = np.zeros_like(layer.weights)
#             layer.bias_momentums = np.zeros_like(layer.biases)
#             layer.bias_cache = np.zeros_like(layer.biases)

#         # zet de momentums
#         layer.weight_momentums = self.beta_1 * \
#             layer.weight_momentums + (1 - self.beta_1) * layer.dweights

#         layer.bias_momentums = self.beta_1 * \
#             layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

#         # correct de momentums
#         weight_momentums_corrected = layer.weight_momentums / \
#             (1 - self.beta_1 ** (self.iterations + 1))

#         bias_momentums_corrected = layer.bias_momentums / \
#             (1 - self.beta_1 ** (self.iterations + 1))

#         # update cache met squared gradients
#         layer.weight_cache = self.beta_2 * layer.weight_cache + \
#             (1 - self.beta_2) * layer.dweights ** 2

#         layer.bias_cache = self.beta_2 * layer.bias_cache + \
#             (1 - self.beta_2) * layer.dbiases ** 2

#         # correct de caches
#         weight_cache_corrected = layer.weight_cache / \
#             (1 - self.beta_2 ** (self.iterations + 1))

#         bias_cache_corrected = layer.bias_cache / \
#             (1 - self.beta_2 ** (self.iterations + 1))

#         layer.weights += -self.current_learning_rate * \
#             weight_momentums_corrected / \
#             (np.sqrt(weight_cache_corrected) + self.epsilon)

#         layer.biases += -self.current_learning_rate * \
#             bias_momentums_corrected / \
#             (np.sqrt(bias_cache_corrected) + self.epsilon)

#     def post_update_params(self):
#         self.iterations += 1


# # Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Dense layer with 2 input features and 64 output values
# dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
#                      bias_regularizer_l2=5e-4)
# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLu()
# # Create second Dense layer with 64 input features (as we take output
# # of previous layer here) and 3 output values (output values)
# dense2 = Layer_Dense(64, 3)
# # Create Softmax classifier's combined loss and activation
# loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
# # Create optimizer
# optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

# for epoch in range(10001):
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     data_loss = loss_activation.forward(dense2.output, y)
#     regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
#         loss_activation.loss.regularization_loss(dense2)
#     loss = data_loss + regularization_loss

#     predictions = np.argmax(loss_activation.output, axis=1)
#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)
#     accuracy = np.mean(predictions == y)

#     if not epoch % 100:
#         print(f'epoch: {epoch}, ' +
#               f'acc: {accuracy:.3f} ' +
#               f'data_loss: {data_loss:.3f} ' +
#               f'reg_loss: {regularization_loss:.3f} ' +
#               f'lr: {optimizer.current_learning_rate}')
#     loss_activation.backward(loss_activation.output, y)
#     dense2.backward(loss_activation.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)

#     optimizer.pre_update_params()
#     optimizer.update_params(dense1)
#     optimizer.update_params(dense2)
#     optimizer.post_update_params()


















# # Validate the model

# # Create test dataset
# X_test, y_test = spiral_data(samples=100, classes=3)

# # Perform a forward pass of our testing data through this layer
# dense1.forward(X_test)

# # Perform a forward pass through activation function
# # takes the output of first dense layer here
# activation1.forward(dense1.output)

# # Perform a forward pass through second Dense layer
# # takes outputs of activation function of first layer as inputs
# dense2.forward(activation1.output)

# # Perform a forward pass through the activation/loss function
# # takes the output of second dense layer here and returns loss
# loss = loss_activation.forward(dense2.output, y_test)

# # Calculate accuracy from output of activation2 and targets
# # calculate values along first axis
# predictions = np.argmax(loss_activation.output, axis=1)
# if len(y_test.shape) == 2:
#     y_test = np.argmax(y_test, axis=1)
# accuracy = np.mean(predictions==y_test)

# print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')