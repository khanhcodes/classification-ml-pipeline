#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

# Train the model based on training data and return the model
def train_SVM(training_data):
    df = training_data
    X_train = df.iloc[: , :-1]
    y_raw = df.iloc[: , -1:]
    y_train = np.ravel(y_raw)
    #print(X_train)
    #print(y_train)
    sm = SMOTE(random_state=1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    svm = SVC()
    svm_ovr = OneVsRestClassifier(svm)
    svm_ovr.fit(X_train_res, y_train_res)