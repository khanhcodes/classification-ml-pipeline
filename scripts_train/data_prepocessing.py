#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import pandas as pd

# Preprocess and return a dataframe of training datac
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
    print('Completed preprocessing data!') 
    return merged_df

