# import matplotlib.pyplot as plt
# import nnfs
# from nnfs.datasets import vertical_data

# nnfs.init()

# X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()


# X, y = spiral_data(samples=100, classes=3)
# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLu()

# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()

# dense1.forward(X)
# activation1.forward(dense1.output)
# dense2.forward(activation1.output)
# activation2.forward(dense2.output)

# loss_function = Loss_CategoricalCrossentropy()
# loss = loss_function.calculate(activation2.output, y)

# predictions = np.argmax(activation2.output, axis=1)

# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)

# accuracy = np.mean(predictions == y)
# print(accuracy)