# shared training and evaluation functions

import numpy as np
from sklearn import linear_model, metrics

def logistic_regression(X_train, X_test, y_train, y_test):
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    y_pp = model.predict_proba(X_test)  # predicted probabilities
    print_metrics(y_test, y_pp)

def print_metrics(y_true, y_pp):
    """
    y_true: test set labels
    y_pp: predicted probabilities
    """
    y_pl = np.argmax(y_pp, axis=1)  # predicted labels
    print("accuracy: ", metrics.accuracy_score(y_true, y_pl))
    print("log loss: ", metrics.log_loss(y_true, y_pp))
