#!/usr/bin/python

import csv
import numpy as np
from pearson import pearson
from CollaborativeFiltering import predict
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

# convert ratings to utility matrix
mat = np.zeros(dtype=float, shape=(n_users, n_items))
with open('../datasets/train_set.csv', 'r') as data:
    next(data)
    for line in csv.reader(data):
        mat[users_mapping[int(line[0])]][items_mapping[int(line[1])]] = float(line[2])

# compute the average rating for all users
avg = []
for i in range(len(mat)):
    avg.append(np.sum(mat[i]) / len(np.where(mat[i] > 0)[0]))

# compute the pearson matrix
sim = pearson(mat, avg)

# compute the prediction ratings
pre = predict(sim, mat, avg, 16)

# recommend k (k=8 here) highest prediction items to each user and compute the sse
evaluate(pre, 8, users_mapping, items_mapping, '../result/collaborative-filtering')
