#!/usr/bin/env python
# coding: utf-8

##This script is to classify unknown seedling data into their cell types based on PC values

##BUILT-IN MODULES
import os
import argparse
import sys
import time
import subprocess

def get_parsed_args():

    parser = argparse.ArgumentParser(description="Cell-type Classification from Crop Plant Seedling Data")
    ##require files
    parser.add_argument("-d", dest='working_dir', default="./", help="Working directory to store intermediate files of "
                                                                     "each step. Default: ./ ")

    parser.add_argument("-o", dest='output_dir', default="./", help="Output directory to store the output files. "
                                                                    "Default: ./ ")
    
    parser.add_argument("-m", dest='model_name', help="Provide one of model names: -m P or -m M or -m F or -m O or -m U."
                                                      "This argument will directly download the model dir,"
                                                      "So users do not need to initiate -m_dir.")
    
    ##parse of parameters
    args = parser.parse_args()
    return args