#!/usr/local/bin/python3
#
# training pipeline using only on phone_brand_id and device_model_id

import numpy as np
import pandas as pd
import pickle
import itertools
import scipy.sparse
from sklearn import cross_validation
import preprocessing as prep
import train

PICKLE_FILE = "../processed_data/pickle/data.pickle"

def prep_data(n=None, test_size=0.2):
    """
    read the data and return as X, y.
    Only use features phone_brand_id and device_model_id
    n: number of rows to read
    test_size: ratio of test size
    """
    df = pickle.load(open(PICKLE_FILE, "rb"))
    if n is not None:
        df = df.head(n=n)
    features =  df.as_matrix(columns=["phone_brand_id", "device_model_id"])
    X = prep.encode_features(features)
    categories = df["label_ids"].tolist()
    M = encode_bag_of_features(categories)
    # concatenate features
    X = np.concatenate((X, M), axis=1)
    y = df["group_id"].tolist()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def encode_bag_of_features(feature):
    """
    Similar to one hot encoding but used to encode a set of
    categorical feature values.
    feature: nested list of one feature column
    Example input: test = np.array([[], [1], [3], [2,3]])
    Expected output:
        [[ 0.  0.  0.  0.]
        [ 0.  1.  0.  0.]
        [ 0.  0.  0.  1.]
        [ 0.  0.  1.  1.]]
    """
    flattened = np.fromiter(itertools.chain.from_iterable(feature), np.int64)
    data = np.ones(flattened.size)
    indices = flattened
    indptr = np.zeros(len(feature)+1, dtype=np.int64)
    for i, row in enumerate(feature):
        indptr[i+1] = indptr[i] + len(row)
    matrix = scipy.sparse.csr_matrix((data, indices, indptr))
    return matrix.todense()

X_train, X_test, y_train, y_test = prep_data()
train.logistic_regression(X_train, X_test, y_train, y_test)
