import numpy as np
from nnfs.datasets import spiral_data
from venumtorchflow.nn.optimizers import Optimizer_Adam
from venumtorchflow.nn.layers import (Layer_Dense, Layer_Dropout)
from venumtorchflow.nn.activations import (Activation_ReLu,
                                        Activation_Softmax_Loss_CategoricalCrossEntropy)


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4)
activation1 = Activation_ReLu()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f} ' +
              f'loss: {loss:.3f} '
              f'data_loss: {data_loss:.3f} ' +
              f'reg_loss: {regularization_loss:.3f} ' +
              f'lr: {optimizer.current_learning_rate}')
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
