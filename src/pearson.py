import numpy as np
from math import sqrt, pow


def pearson_calc(x, y, avg_x, avg_y):
    """
    :param x: vector x
    :param y: vector y
    :param avg_x: the average of vector x
    :param avg_y: the average of vector y
    :return: the pearson similarity of vector x and vector y
    """
    sum_xy = 0.
    root_x = 0.
    root_y = 0.
    product = x * y

    # if no valid data
    if not np.sum(product):
        return 0

    # compute the pearson similarity
    for i in list(np.where(product > 0)[0]):
        if x[i] and y[i]:
            sum_xy += (x[i] - avg_x) * (y[i] - avg_y)
            root_x += pow(x[i] - avg_x, 2)
            root_y += pow(y[i] - avg_y, 2)
    root_x = sqrt(root_x)
    root_y = sqrt(root_y)

    # avoid division by 0
    if root_x and root_y:
        return sum_xy / (root_x * root_y)
    else:
        return 1


def pearson(mat, avg):
    """
    :param mat: the util matrix
    :param avg: the average of all vectors in the util matrix
    :return: the pearson matrix
    """
    sim = np.zeros(dtype=float, shape=(len(mat), len(mat)))

    # compute the pearson similarities for each pair of users
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            sim[i][j] = pearson_calc(mat[i], mat[j], avg[i], avg[j])
            sim[j][i] = sim[i][j]

    return sim
