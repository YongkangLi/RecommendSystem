#!/usr/bin/python

import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ContentBased import predict
from evaluate import evaluate


users = set()           # a record for all users
items = set()           # a record for all items
features = set()        # a record for all tags
users_mapping = {}      # mapping from user-id to utility matrix index
items_mapping = {}      # mapping from item-id to utility matrix index
features_mapping = {}   # mapping from tag to matrix index
characteristics = {}    # a record of all tags of all items

# collect all users and items
with open('../datasets/train_set.csv', 'r') as data:
    next(data)
    for line in csv.reader(data):
        users.add(int(line[0]))
        items.add(int(line[1]))

# collect all tags and tags of all items
with open('../datasets/movies.csv', 'r') as data:
    next(data)
    for line in csv.reader(data):
        items.add(int(line[0]))
        new = line[2].split('|')
        characteristics[int(line[0])] = new
        for feature in new:
            features.add(feature)

# install mapping from id to matrix index
i = 0
for user in users:
    users_mapping[user] = i
    i += 1
i = 0
for item in items:
    items_mapping[item] = i
    i += 1

# install mapping from tag to matrix index
i = 0
for feature in features:
    features_mapping[feature] = i
    i += 1

n_users = len(users_mapping)
n_items = len(items_mapping)
n_features = len(features_mapping)

# convert ratings to utility matrix
mat = np.zeros(dtype=float, shape=(n_users, n_items))
freq = np.zeros(dtype=float, shape=(n_items, n_features))
with open('../datasets/train_set.csv', 'r') as data:
    next(data)
    for line in csv.reader(data):
        mat[users_mapping[int(line[0])]][items_mapping[int(line[1])]] = float(line[2])

# convert tags to frequency matrix
for key in characteristics.keys():
    for characteristic in characteristics[key]:
        freq[items_mapping[key]][features_mapping[characteristic]] = 1

# compute tf-idf
transformer = TfidfTransformer()
tf_idf = transformer.fit_transform(freq).toarray()

# compute the cosine similarity
similarity = cosine_similarity(tf_idf)

# compute the prediction ratings
pre = predict(similarity, mat)

# recommend k (k=8 here) highest prediction items to each user and compute the sse
evaluate(pre, 8, users_mapping, items_mapping, '../result/content-based')
