#!/usr/bin/env python
# coding: utf-8

##This script is to train different models for classifying unknown seedling data into their cell types 

## BUILT-IN MODULES
import os
import argparse
from pyexpat import model
import sys
import time
import pickle 

from scripts_train.data_prepocessing import preprocess_data, preprocess_indep_test
from scripts_train.model_training import train_model, validation_curve

# Prepocess training data

def get_parsed_args():

    parser = argparse.ArgumentParser(description="Crop Plant Seedling Classification Based On PC Values")
    
    ## Required arguments
    # Take in the path of raw training_embeddings file from user
    parser.add_argument("-d", dest='working_dir', default="./", help="Working directory to store intermediate files of "
                                                                     "each step. Default: ./ ")

    parser.add_argument("-o", dest='output_dir', default="./", help="Output directory to store the output files. "
                                                                    "Default: ./ ")
    
    parser.add_argument("-train", dest= 'train_data_location', help="Provide the relative path to the location of training embeddings file.", 
                        type=str)

    parser.add_argument("-meta", dest= 'meta_train_location', help="Provide the relative path to the location of meta train file.", 
                        type=str)
    
    parser.add_argument("-test", dest= 'test_data_location', help="Provide the relative path to the location of independent testing embeddings file.", 
                        type=str)

    parser.add_argument("-meta_t", dest= 'meta_test_location', help="Provide the relative path to the location of meta test file.", 
                        type=str)
    
    # Take in a command-line arguement specifying the model the user wants to train
    parser.add_argument("-model", dest="model_name", help="Provide the name of the classifier you want to train.", 
                        type=str)

    args = parser.parse_args()
    return args

def main(argv=None):

    if argv is None:
        argv = sys.argv
    args = get_parsed_args()

    ######################################
    output_dir = args.output_dir
    if not output_dir.endswith('/'):
        output_dir = output_dir + '/'
    else:
        output_dir = output_dir
    
    ##Check whether the files are provided
    if args.train_data_location is not None:
        train_embeddings = args.train_data_location
    else: 
        print("Please input the relative path to the training embeddings file!")
        
    if args.meta_train_location is not None:
        train_meta = args.meta_train_location
    else: 
        print("Please input the relative path to the meta train file!")
        
    if args.test_data_location is not None:
        test_embeddings = args.test_data_location
    else: 
        print("Please input the relative path to the testing embeddings file!")
        
    if args.meta_test_location is not None:
        test_meta = args.meta_test_location
    else: 
        print("Please input the relative path to the meta test file!")
    
    # Prepocess training data
    df_train = preprocess_data(train_embeddings, train_meta)
    #df_test = preprocess_indep_test(test_embeddings, test_meta)

    if args.model_name is not None:
        model_name = args.model_name
        if model_name != 'LR' and model_name != 'KNN' and model_name != 'RF' and model_name != 'SVM':
            print("Please use one of 'LR', 'KNN', 'RF', 'SVM' to be model name")
        else: 
            #train_model(df_train, model_name, output_dir)
            validation_curve(df_train, model_name, output_dir)
            
    else: 
        print("Please input a model name!")

if __name__ == "__main__":

    start_time = time.time()
    print('start time is ' + str(start_time))
    main()

    print("--- %s seconds ---" % (time.time() - start_time))
    

    







