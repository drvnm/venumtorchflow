import math
import numpy as np
# softmax_output = [0.7, 0.1, 0.2]
# target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0]) * target_output[0] +
#          math.log(softmax_output[1]) * target_output[1] +
#          math.log(softmax_output[2]) * target_output[2])

# loss_one = -(math.log(softmax_output[0]) * target_output[0])
# print(loss)
# print(loss_one)


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4, ],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])


# alleen de correcte indexes in een array
if len(class_targets.shape) == 1:
    # alle hoogste scores
    correct_confidence = softmax_outputs[range(len(softmax_outputs)), class_targets]
    print(correct_confidence)

elif len(class_targets.shape) == 2:
    correct_confidence = np.sum(softmax_outputs * class_targets, axis=1)

losses = -np.log(correct_confidence)
avg_loss = np.mean(losses)
print(correct_confidence)