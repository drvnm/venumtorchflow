import numpy as np

outputs = np.array([[1,  2,  3],
                    [2,  4,  6],
                    [0,  5,  10],
                    [11, 12, 13],
                    [5,  10, 15]])


sample_losses = np.mean(outputs, axis=-1)
print(sample_losses)