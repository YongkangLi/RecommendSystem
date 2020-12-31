import numpy as np


def predict(sim, mat, avg, k):
    """
    :param sim: the similarities matrix
    :param mat: the utility matrix
    :param avg: the average of all vectors in the utility matrix
    :param k: the top k highest to pick
    :return: the prediction ratings
    """
    prediction = np.zeros(dtype=float, shape=mat.shape)
    for i in range(mat.shape[0]):
        for j in range(sim.shape[0]):
            if mat[i][j] == 0:
                # filter those users without ratings for the jth item
                product = (sim[i] * mat[:, [j]].T[0])
                candidates = np.argwhere(mat.T[j] > 0).T[0]

                # pick the top k highest similarity users
                top_k = (-sim[i][candidates]).argsort()[:k]
                indexes = candidates[top_k]

                # compute the average according to the weight of similarities
                sum_w = np.sum(sim[i][indexes])
                prediction[i][j] = avg[i] if sum_w == 0 else np.sum(product[indexes]) / sum_w
    return prediction
