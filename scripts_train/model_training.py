#!/usr/bin/env python
# coding: utf-8

import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import numpy as np
from numpy import mean
import pickle

# Train the model based on training data and return the model
def train_model(training_data, model_name, output_dir):
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
        cross_validate(df, logReg_model)
        
        
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

def save_model(path, filename):
    fullpath = os.path.join(path, filename)
    print("The complete path to the saved model is ", fullpath)
    
#This method performs a sensitivity analysis of k values and k-fold cross validation
def cross_validate(training_data, model):
    df = training_data
    X_train = df.iloc[: , :-1]
    y_raw = df.iloc[: , -1:]
    y_train = np.ravel(y_raw)
    
    print("----------------------------------------------")
    print("Started configuring k-fold cross validation...")
    folds = range(2, 4)
    means, mins, maxs = list(),list(),list()

    for k in folds:
        cv = KFold(n_splits=k, shuffle=True, random_state=1)
        scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=5)
        k_mean, k_min, k_max = mean(scores), scores.min(), scores.max()
        print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
        means.append(k_mean)
        if max(means) == k_mean:
            optimal_k = k
	    # store min and max relative to the mean
        mins.append(k_mean - k_min)
        maxs.append(k_max - k_mean)
        # line plot of k mean values with min/max error bars
    pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
    pyplot.savefig("mean_accuracy_k_fold_cross_validation.png")
    print("Saved the line plot of mean accuracy for k-fold cross validation " +  
          "in the working directory successfully!")
    print("Chosing k =", optimal_k, " for cross validation...")
    
    kfold = KFold(n_splits=optimal_k, shuffle=True, random_state=1)
    #train = train index, test = test index
    i = 0
    acc = []
    for train, test in kfold.split(X_train, y_train):
        #get the xtrain, ytrain, ytest and xtest data based on the indices
        Xtrain = X_train.iloc[train] 
        ytrain = y_train[train]
        Xtest = X_train.iloc[test]
        ytest = y_train[test]
        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xtest)
        acc_k = accuracy_score(ytest, y_pred)
        acc.append(acc_k)
        i += 1
        print("Accuracy for fold", i, ":", acc_k)
    print("Mean accuracy: ", mean(acc))
    pickle.dump(model, open('LR_model.pkl', 'wb'))
    print("Saved model successfully!")
    
    
    