#!/usr/bin/env python
# coding: utf-8

##This script is to train different models for classifying unknown seedling data into their cell types 

##BUILT-IN MODULES
import os
import argparse
import sys
import time

from sympy import re

from scripts_train.data_prepocessing import preprocess_data
from scripts_train.LR_training import train_LR
from scripts_train.KNN_training import train_KNN


df_train = preprocess_data('data\opt_training_embeddings.csv', 'data\opt_meta_train.csv')

def get_parsed_args():

    parser = argparse.ArgumentParser(description="Crop Plant Cell-type Classification Model Training")
    ##require files

    parser.add_argument("-m", dest='model_name', help="Provide one of model names: -m LR or -m KNN or -m RF or -m SVM."
                                                      "This argument will train the corresponding model.")
    
    ##parse of parameters
    args = parser.parse_args()
    return args

def main(argv=None):

    if argv is None:
        argv = sys.argv
    args = get_parsed_args()

    ######################################
    
    if args.model_name is not None:
        model_name = args.model_name
        if model_name != 'LR' and model_name != 'KNN' and model_name != 'RF' and model_name != 'SVM':
            print("Please use one of 'LR', 'KNN', 'RF', 'SVM' to be model name")
            return
        else:
            if model_name == 'LR':
                

            if model_name == 'M':
                ##create a dir in the working_dir to store the model dir
                google_drive_path = '1ExRwC3szJ4XMa3ikxM9Ccu31lY79rdw9'
                download_model_dir = download_model(working_dir, model_name, google_drive_path)

            if model_name == 'F':
                ##create a dir in the working_dir to store the model dir
                google_drive_path = '1uvnm99ypauIKtqCxoybdtT-mEMdoupip'
                download_model_dir = download_model(working_dir, model_name, google_drive_path)

            if model_name == 'O':
                ##create a dir in the working_dir to store the model dir
                google_drive_path = '1Q6HW1NhNs0a6Ykrw7jGEKKPWxawpWiuM'
                download_model_dir = download_model(working_dir, model_name, google_drive_path)

            if model_name == 'U':
                ##create a dir in the working_dir to store the model dir
                google_drive_path = '1uXTEtNQtJc2DO-JpT0s4Kv1k2ogUjCLr'
                download_model_dir = download_model(working_dir, model_name, google_drive_path)




