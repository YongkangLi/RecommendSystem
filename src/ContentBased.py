import numpy as np


def predict(sim, util):
    """
    :param sim: the similarities matrix
    :param util: the utility matrix
    :return: the prediction ratings
    """
    prediction = np.zeros(dtype=float, shape=util.shape)
    for i in range(util.shape[0]):
        for j in range(sim.shape[0]):
            if util[i][j] == 0:
                product = util[i] * sim[j]

                # filter those unrated
                indexes = np.argwhere(product > 0)

                # compute the average according to the weight of similarities
                sum_w = np.sum(sim[j][indexes])
                prediction[i][j] = 2.5 if sum_w == 0 else np.sum(product[indexes]) / sum_w

    return prediction
