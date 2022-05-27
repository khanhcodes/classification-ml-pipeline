#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

# Train the model based on training data and return the model
def train_model(training_data, model_name):
    df = training_data
    X_train = df.iloc[: , :-1]
    y_raw = df.iloc[: , -1:]
    y_train = np.ravel(y_raw)
    
    # Use SMOTE to fix undersampling of minority classes
    sm = SMOTE(random_state=1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # Train the model based on the corresponding model name the user inputs
    if model_name == "LR":
        logReg_model = LogisticRegression(multi_class='ovr')
        print("Training the model...")
        logReg_model.fit(X_train_res, y_train_res)
        print("Completed training logistic regression model!")
        return logReg_model
    elif model_name == "KNN":
        knn_model = KNeighborsClassifier(n_neighbors=10)
        knn_ovr = OneVsRestClassifier(knn_model)
        print("Training the model...")
        knn_ovr.fit(X_train_res, y_train_res)
        print("Completed training KNN model!")
        return knn_ovr
    elif model_name == "RF":
        rf = RandomForestClassifier()
        rf_ovr = OneVsRestClassifier(rf)
        print("Training the model...")
        rf_ovr.fit(X_train_res, y_train_res)
        print("Completed training Random Forest model!")
        return rf_ovr
    else: 
        svm = SVC()
        svm_ovr = OneVsRestClassifier(svm)
        print("Training the model...")
        svm_ovr.fit(X_train_res, y_train_res)
        print("Completed training SVM model!")
        return svm_ovr

    
        
        
        
