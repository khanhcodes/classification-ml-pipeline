#!/usr/bin/env python
# coding: utf-8

import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from yellowbrick.model_selection import ValidationCurve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import numpy as np
from numpy import mean
import joblib
from pathlib import Path

root = Path(".")

# Train the model based on training data and return the model
def train_model(training_data, model_name, output_dir, working_dir):
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
        temp_model = model_name + "_model.sav"
        filename = working_dir + temp_model
        joblib.dump(logReg_model, filename)
        print("Completed training logistic regression model!")
        cross_validate(df, filename, output_dir, model_name)
    
    elif model_name == "KNN":
        knn_model = KNeighborsClassifier(n_neighbors=10)
        knn_ovr = OneVsRestClassifier(knn_model)
        print("Training the model...")
        knn_ovr.fit(X_train_res, y_train_res)
        print("Completed training KNN model!")
        cross_validate(df, knn_ovr, output_dir, model_name)
        
    elif model_name == "RF":
        rf = RandomForestClassifier()
        rf_ovr = OneVsRestClassifier(rf)
        print("Training the model...")
        rf_ovr.fit(X_train_res, y_train_res)
        print("Completed training Random Forest model!")
        cross_validate(df, rf_ovr, output_dir, model_name)
        
    else: 
        svm = SVC()
        svm_ovr = OneVsRestClassifier(svm)
        print("Training the model...")
        svm_ovr.fit(X_train_res, y_train_res)
        print("Completed training SVM model!")
        cross_validate(df, svm_ovr, output_dir, model_name)

def save_model(path, filename):
    fullpath = os.path.join(path, filename)
    print("The complete path to the saved model is ", fullpath)
    
# Perform a sensitivity analysis of k values and k-fold cross validation
def cross_validate(training_data, temp_model_file, output_dir, name):
    df = training_data
    X_train = df.iloc[: , :-1]
    y_raw = df.iloc[: , -1:]
    y_train = np.ravel(y_raw)
    
    print("----------------------------------------------")
    print("Started configuring k-fold cross validation...")
    folds = range(2, 11)
    means, mins, maxs = list(),list(),list()

    temp_model = joblib.load(temp_model_file)
    for k in folds:
        cv = KFold(n_splits=k, shuffle=True, random_state=1)
        scores = cross_val_score(temp_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=5)
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
    pyplot.xlabel("Fold")
    pyplot.ylabel("Mean accuracy score")
    pyplot.title("Mean accuracy for each fold line plot")
    plt_filename = output_dir + "/graphs/mean_accuracy_k_fold_cross_validation.png"
    pyplot.savefig(plt_filename)
    print("Saved the line plot of mean accuracy for k-fold cross validation " +  
          "in the output directory successfully!")
    print("Chosing k =", optimal_k, " for cross validation...")
    
    kfold = KFold(n_splits=optimal_k, shuffle=True, random_state=1)
    i = 0
    acc = []
    # train = train index, test = test index
    for train, test in kfold.split(X_train, y_train):
        #get the xtrain, ytrain, ytest and xtest data based on the indices
        Xtrain = X_train.iloc[train] 
        ytrain = y_train[train]
        Xtest = X_train.iloc[test]
        ytest = y_train[test]
        temp_model.fit(Xtrain, ytrain)
        y_pred = temp_model.predict(Xtest)
        acc_k = accuracy_score(ytest, y_pred)
        acc.append(acc_k)
        i += 1
        print("Accuracy for fold", i, ":", acc_k)
    print("Mean accuracy: ", mean(acc))
    model_name = "/" + str(name) + "_finalized_model.sav"
    filename = output_dir + model_name
    joblib.dump(temp_model, filename)
    print("Saved retrained model successfully!")
    
# Perform hyperparameter tuning for optimal parameter
#def param_tuning(training_data, model_name, output_dir):
    
    
# Graph the validation curve for hyperparameter tuning
def validation_curve(training_data, model_name, output_dir):
    model = str(model_name)
    
    # Split training data
    df = training_data
    X_train = df.iloc[: , :-1]
    y_raw = df.iloc[: , -1:]
    y_train = np.ravel(y_raw)
    
    if model == "LR":
        clf = LogisticRegression(multi_class='ovr')
        param_name = "C"
        param_range=np.logspace(-6, 8, 8)
    elif model == "KNN":
        clf = KNeighborsClassifier()
        param_name = "n_neighbors"
        param_range = np.arange(1, 30, 1)
    elif model == "RF":
        clf = RandomForestClassifier()
        param_name = "max_depth"
        param_range = np.arange(1, 20, 1)
    else:
        clf = SVC()
        param_name="gamma"
        param_range = np.logspace(-6, -1, 12)
        
    print("------------------------------------------------")
    print("Graphing the validation curve for " + model + " model...")
    if model == "LR" or "SVM":
        crossval_lr = ValidationCurve(clf, param_name, param_range, logx=True, cv=5, n_jobs=8)
    else:
        crossval_lr = ValidationCurve(clf, param_name, param_range, logx=False, cv=5, n_jobs=8)
    crossval_lr.fit(X_train, y_train)
    crossval_lr.show(outpath=output_dir + "graphs/" + model + "_crossval.png")
    print("Saved the validation curve successfully in output directory!")
        

        
    
    
    