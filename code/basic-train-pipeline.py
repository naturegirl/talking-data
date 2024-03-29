#!/usr/local/bin/python3
#
# training pipeline using only on phone_brand_id and device_model_id

import numpy as np
import pandas as pd
import pickle
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
    y = df["group_id"].tolist()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prep_data()
train.logistic_regression(X_train, X_test, y_train, y_test)
