#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

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
        pickle.dump(logReg_model, open('LR_model.pkl', 'wb'))
        print("Completed training logistic regression model!")
        print("Your model is now saved in the folder as a pickle file.")
    elif model_name == "KNN":
        knn_model = KNeighborsClassifier(n_neighbors=10)
        knn_ovr = OneVsRestClassifier(knn_model)
        print("Training the model...")
        knn_ovr.fit(X_train_res, y_train_res)
        pickle.dump(knn_ovr, open('KNN_model.pkl', 'wb'))
        print("Completed training KNN model!")
        print("Your model is now saved in the folder as a pickle file.")
    elif model_name == "RF":
        rf = RandomForestClassifier()
        rf_ovr = OneVsRestClassifier(rf)
        print("Training the model...")
        rf_ovr.fit(X_train_res, y_train_res)
        pickle.dump(rf_ovr, open('RF_model.pkl', 'wb'))
        print("Completed training Random Forest model!")
        print("Your model is now saved in the folder as a pickle file.")
    else: 
        svm = SVC()
        svm_ovr = OneVsRestClassifier(svm)
        print("Training the model...")
        svm_ovr.fit(X_train_res, y_train_res)
        pickle.dump(svm_ovr, open('SVM_model.pkl', 'wb'))
        print("Completed training SVM model!")
        print("Your model is now saved in the folder as a pickle file.")

    
        
        
        
