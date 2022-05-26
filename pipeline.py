#!/usr/bin/env python
# coding: utf-8

##BUILT-IN MODULES
import os
import argparse
import sys
import time
import subprocess

def get_parsed_args():

    parser = argparse.ArgumentParser(description="Cell-type Classification for Crop Plants Seedling Data")
    ##require files
    parser.add_argument("-d", dest='working_dir', default="./", help="Working directory to store intermediate files of "
                                                                     "each step. Default: ./ ")

    parser.add_argument("-o", dest='output_dir', default="./", help="Output directory to store the output files. "
                                                                    "Default: ./ ")
    ##parse of parameters
    args = parser.parse_args()
    return args