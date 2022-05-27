#!/usr/bin/env python
# coding: utf-8

##This script is to train different models for classifying unknown seedling data into their cell types 

## BUILT-IN MODULES
import os
import argparse
import sys
import time
from tokenize import String

from scripts_train.data_prepocessing import preprocess_data
from scripts_train.model_training import train_model

parser = argparse.ArgumentParser()

# Take in the path of raw training_embeddings file from user
parser.add_argument("train_data_location", help="Provide the relative path to the location of training embeddings file.", 
                    type=str)
args = parser.parse_args()
if args.train_data_location is not None:
    train_embeddings = args.train_data_location
else: 
    print("Please input the relative path to the training embeddings file!")

# Take in the path of raw meta_train file from user
parser.add_argument("meta_train_location", help="Provide the relative path to the location of meta train file.", 
                    type=str)
args = parser.parse_args()
if args.meta_train_location is not None:
    train_meta = args.meta_train_location
else: 
    print("Please input the relative path to the meta train file!")
    
# Prepocess training data
df_train = preprocess_data(train_embeddings, train_meta)

# Take in a command-line arguement specifying the model the user wants to train

parser.add_argument("model", help="Provide the name of the classifier you want to train.", 
                    type=str)
args = parser.parse_args()

if args.model is not None:
    model_name = args.model
    if model_name != 'LR' and model_name != 'KNN' and model_name != 'RF' and model_name != 'SVM':
        print("Please use one of 'LR', 'KNN', 'RF', 'SVM' to be model name")
    else: 
        train_model(df_train, model_name)
else: 
    print("Please input a model name!")

