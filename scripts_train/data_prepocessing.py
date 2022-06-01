#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# Preprocess and return a dataframe of training data
def preprocess_data(training_embeddings, meta_train):
    # Transpose the training data
    df_raw = pd.read_csv(training_embeddings, index_col=0)
    df = df_raw.T

    # Remove any inconsistency in the cell name in training data
    meta_train_raw_df = pd.read_csv(meta_train)
    meta_train_processed_1= meta_train_raw_df.replace('-','.', regex=True)
    meta_train_processed_2= meta_train_processed_1.replace(':','.', regex=True)
    meta_train_processed_2.to_csv('data\opt_meta_train_processed.csv')
    
    # Add the cell type next to cell name 
    df_processed = pd.read_csv ('data\opt_meta_train_processed.csv', index_col=0)
    cell_type_df = df_processed['cell_type']
    df.index = pd.RangeIndex(len(df))
    cell_type_df = pd.RangeIndex(len(cell_type_df))
    merged_df = df.join(df_processed['cell_type'])
    merged_df.to_csv('data\opt_training_embeddings_processed.csv')
    print('Completed preprocessing training data!') 
    return merged_df

# Preprocess and return a dataframe of testing data
def preprocess_indep_test(indep_testing_embeddings, meta_test):
    # Transpose the testing data
    df_raw = pd.read_csv(indep_testing_embeddings, index_col=0)
    df = df_raw.T
    
    # Take the cell-type column from the meta test file and 
    # add to the testing embeddings to create a testing dataset
    meta_test_raw_df = pd.read_csv (meta_test)
    cell_type_df = meta_test_raw_df['cell_type']
    df.index = pd.RangeIndex(len(df))
    cell_type_df = pd.RangeIndex(len(cell_type_df))
    merged_df = df.join(meta_test_raw_df['cell_type'])
    merged_df.to_csv('data\opt_indep_testing_embeddings_processed.csv')
    print('Completed preprocessing testing data!') 
    return merged_df

