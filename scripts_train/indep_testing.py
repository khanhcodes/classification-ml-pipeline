#!/usr/bin/env python
# coding: utf-8

import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Test the model on independent testing data and get the accuracy 
def indep_test(testing_data, model_name, output_dir):
    df = testing_data
    X_test = df.iloc[: , :-1]
    y_test = df.iloc[: , -1:]
    y_test = np.ravel(y_test)
    
    model = str(model_name)
    if model == "LR":
        temp_model = model + "_finalized_model.sav"
        filename = output_dir + temp_model
        
    # Load model from disk
    print("---------------------------------------------")
    print("Predicting cell-type on the independent testing data...")
    clf = joblib.load(filename)
    y_pred = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    
    
    