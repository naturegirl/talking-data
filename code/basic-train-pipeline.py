#!/usr/local/bin/python3
#
# training pipeline using only on phone_brand_id and device_model_id

import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
from sklearn import linear_model, metrics
from sklearn.preprocessing import OneHotEncoder

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
    X = encode_features(features)
    y = df["group_id"].tolist()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def encode_features(features):
    """
    Use one hot encoding to encode the 2 categorical features
    phone_brand_id and device_model_id
    """
    enc = OneHotEncoder()
    enc.fit(features)
    return enc.transform(features).toarray()

def print_metrics(y_true, y_pp):
    """
    y_true: test set labels
    y_pp: predicted probabilities
    """
    y_pl = np.argmax(y_pp, axis=1)  # predicted labels
    print("accuracy: ", metrics.accuracy_score(y_test, y_pl))
    print("log loss: ", metrics.log_loss(y_test, y_pp))

def logistic_regression(X_train, X_test, y_train, y_test):
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    y_pp = model.predict_proba(X_test)  # predicted probabilities
    print_metrics(y_test, y_pp)

X_train, X_test, y_train, y_test = prep_data()
logistic_regression(X_train, X_test, y_train, y_test)
