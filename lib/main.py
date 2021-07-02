# import numpy as np
# import os
# import cv2
# from tkinter import *

# import venumtorchflow.nn as vtf


# # model = vtf.Model.load('fmnist2.model')
# # print(model)
# # Loads a MNIST dataset
# def load_mnist_dataset(dataset, path):

#     # Scan all the directories and create a list of labels
#     labels = os.listdir(os.path.join(path, dataset))

#     # Create lists for samples and labels
#     X = []
#     y = []

#     # For each label folder
#     for label in labels:
#         # And for each image in given folder
#         for file in os.listdir(os.path.join(path, dataset, label)):
#             # Read the image
#             image = cv2.imread(
#                 os.path.join(path, dataset, label, file),
#                 cv2.IMREAD_UNCHANGED)

#             # And append it and a label to the lists
#             X.append(image)
#             y.append(label)

#     # Convert the data to proper numpy arrays and return
#     return np.array(X, dtype=object), np.array(y).astype('uint8')


# # MNIST dataset (train + test)
# def create_data_mnist(path):

#     # Load both sets separately
#     X, y = load_mnist_dataset('train', path)
#     X_test, y_test = load_mnist_dataset('test', path)

#     # And return all the data
#     return X, y, X_test, y_test


# # Create dataset
# X, y, X_test, y_test = create_data_mnist('demo/fashion_mnist_images')
# # Shuffle the training dataset
# keys = np.array(range(X.shape[0]))
# np.random.shuffle(keys)
# X = X[keys]
# y = y[keys]
# # Scale and reshape samples
# X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
#           127.5) / 127.5


# model = vtf.Model()
# model.add(vtf.Layer_Dense(X.shape[1], 128))
# model.add(vtf.Activation_ReLu())
# model.add(vtf.Layer_Dense(128, 128))
# model.add(vtf.Activation_ReLu())
# model.add(vtf.Layer_Dense(128, 10))
# model.add(vtf.Activation_Softmax())

# model.set(
#     loss=vtf.Loss_CategoricalCrossentropy(),
#     accuracy=vtf.Accuracy_Categorical(),
#     optimizer=vtf.Optimizer_Adam(decay=1e-5),
# )

# model.finalize()

# model.train(X, y, validation_data=(X_test, y_test),
#             epochs=2, batch_size=128, print_every=100)

# model.evaluate(X_test, y_test)
# model.save('fmnist2.model')
import venumtorchflow.nn as vtf


model = vtf.Model.load('fmnist2.model')
print(model)
