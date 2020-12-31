import sys
import numpy as np
import random


def minhash(data, k=10):
    """
    :param data: util matrix
    :param k: times of hashed
    :return: MinHash signatures
    """
    rows, cols = len(data), len(data[0])
    signature = []

    # initialize the matrix with infinite
    for i in range(k):
        signature.append([sys.maxsize] * cols)

    # a sequence of index
    hash_value = list(range(rows))
    for i in range(k):
        # shuffle the sequence (equal to hash)
        random.shuffle(hash_value)
        for c in range(cols):
            # locate the minimum
            for r in range(rows):
                if data[r][c]:
                    if signature[i][c] > hash_value[r]:
                        signature[i][c] = hash_value[r]
    return signature


def jaccard(signatures):
    """
    :param signatures: MinHash signatures
    :return: jaccard similarities matrix
    """
    signatures = signatures.T
    sim = np.zeros(dtype=float, shape=(signatures.shape[0], signatures.shape[0]))

    # let the probability of equality be the approximated jaccard similarity
    for i in range(signatures.shape[0]):
        for j in range(i + 1, signatures.shape[0]):
            sim[j][i] = sim[i][j] = len(set(signatures[i]) & set(signatures[j]))
    return sim / signatures.shape[1]
