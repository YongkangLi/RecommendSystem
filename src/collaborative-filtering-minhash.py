#!/usr/bin/python

import csv
import numpy as np
from CollaborativeFiltering import predict
from MinHash import minhash, jaccard
from evaluate import evaluate


users = set()       # a record for all users
items = set()       # a record for all items
users_mapping = {}  # mapping from user-id to utility matrix index
items_mapping = {}  # mapping from item-id to utility matrix index

# collect all users and items
with open('../datasets/train_set.csv', 'r') as data:
    next(data)
    for line in csv.reader(data):
        users.add(int(line[0]))
        items.add(int(line[1]))

# install mapping from id to matrix index
i = 0
for user in users:
    users_mapping[user] = i
    i += 1
i = 0
for item in items:
    items_mapping[item] = i
    i += 1
n_users = len(users_mapping)
n_items = len(items_mapping)

# convert ratings to utility matrix and normalize it for MinHash
mat = np.zeros(dtype=float, shape=(n_users, n_items))
normalized = np.zeros(dtype=int, shape=(n_users, n_items))
with open('../datasets/train_set.csv', 'r') as data:
    next(data)
    for line in csv.reader(data):
        normalized[users_mapping[int(line[0])]][items_mapping[int(line[1])]] = 0 if float(line[2]) < 3 else 1
        mat[users_mapping[int(line[0])]][items_mapping[int(line[1])]] = float(line[2])

# compute the MinHash signatures
signatures = np.array(minhash(normalized.T, 10))

# compute the jaccard similarities
sim = jaccard(signatures)

# compute the prediction ratings
pre = predict(sim, mat, [2.5 for i in range(mat.shape[0])], 16)

# recommend k (k=8 here) highest prediction items to each user and compute the sse
evaluate(pre, 8, users_mapping, items_mapping, '../result/collaborative-filtering-minhash')
