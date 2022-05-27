#!/usr/bin/env python
# coding: utf-8

from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

# Train the model based on training data and return the model
def train_KNN(training_data):
    df = training_data
    X_train = df.iloc[: , :-1]
    y_raw = df.iloc[: , -1:]
    y_train = np.ravel(y_raw)
    #print(X_train)
    #print(y_train)
    sm = SMOTE(random_state=1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train_res, y_train_res)
    return knn_model