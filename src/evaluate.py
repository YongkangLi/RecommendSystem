import csv
import numpy as np


def sse(pre, users_mapping, items_mapping):
    """
    :param pre: the prediction ratings
    :param users_mapping: the mapping from user-id to matrix index
    :param items_mapping: the mapping from item-id to matrix index
    :return: sse
    """
    error = []
    with open('../datasets/test_set.csv', 'r') as data:
        next(data)
        for line in csv.reader(data):
            error.append(pre[users_mapping[int(line[0])]][items_mapping[int(line[1])]] - float(line[2]))
    return np.sum(np.square(error))


def recommend(pre, k, users_mapping, items_mapping):
    """
    :param pre: the prediction ratings
    :param k: the number of the top highest prediction ratings
    :param users_mapping: the mappings from user-id to matrix index
    :param items_mapping: the mappings from item-id to matrix index
    :return: the top k highest prediction ratings for each user
    """
    items_reverse = {v: k for k, v in items_mapping.items()}
    users_reverse = {v: k for k, v in users_mapping.items()}

    recommendations = {}    # the record for the top k recommendation for each user

    for i in range(pre.shape[0]):
        # pick the top k highest prediction ratings
        top_k = (-pre[i]).argsort()[:k]
        recommendation = []

        # convert the matrix index back to item-id
        for j in range(k):
            recommendation.append(items_reverse[top_k[j]])
        recommendations[users_reverse[i]] = recommendation

    return recommendations


def evaluate(pre, k, users_mapping, items_mapping, file_name):
    """
    :param pre: the prediction ratings matrix
    :param k: the number of the top highest prediction ratings
    :param users_mapping: the mappings from user-id to matrix index
    :param items_mapping: the mappings from item-id to matrix index
    :param file_name: the path of the result
    :return: null
    """

    # compute the sse
    s = sse(pre, users_mapping, items_mapping)

    # pick the top k highest prediction ratings to recommend for each user
    rec = recommend(pre, k, users_mapping, items_mapping)

    # save the result
    print(s)
    f = open(file_name, 'w')
    for item in rec.items():
        k, v = item
        f.write(str(k) + ' : ' + str(v) + '\n')
