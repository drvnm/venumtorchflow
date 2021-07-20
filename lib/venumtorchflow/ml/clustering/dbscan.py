# import numpy as np


# class DBSCAN:
#     def __init__(self, eps=1, min_points=5):
#         # eps is de "radius" waarin elk punt moet liggen
#         self.eps = eps
#         # min points is het minimale aantal punten
#         # wat in een radius moet liggen om core point te zijn.
#         self.min_points = min_points

#     def train(self, data):
#         point_label = [0] * len(data)
#         point_count = []

#         core = []
#         border = []

#         for i in range(len(data)):
#             point_count_list = self.neighbor_points(data, i)
#             point_count.append(point_count_list)

#         for i in range(len(point_count)):
#             if len(point_count[i]) >= self.min_points:
#                 point_label[i] = -1
#                 core.append(i)
#             else:
#                 border.append(i)
#         for i in border:
#             for j in point_count[i]:
#                 if j in core:
#                     point_label[i] = -2
#                     break

#     def neighbor_points(self, data, point):
#         points = []
#         for i in range(len(data)):
#             distance = np.linalg.norm(data[i] - point)
#             if distance <= self.eps:
#                 points.append(i)
