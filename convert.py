#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 14:04.

@author: a.scotto

Description:
"""

import os
import pickle
import fnmatch
import argparse

from utils.operator import extract

OPERATORS_PATH = 'operators/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder',
                    default=OPERATORS_PATH,
                    help='Path to folder containing MATLAB format stored matrices.')

args = parser.parse_args()

# Filter the content of the folder to select only the .mat files
operators_files = fnmatch.filter(os.listdir(args.folder), '*.mat')

for operator_file_name in operators_files:

    # Skip .mat files containing SVD decomposition
    if operator_file_name.endswith('_SVD.mat'):
        continue

    operator_path = os.path.join(args.folder, operator_file_name)

    # Extract content
    print('Extracting from {}... '.format(operator_path), end='')
    operator = extract(operator_path)
    print('Done.')

    with open('operators/' + operator['name'], 'wb') as file:
        p = pickle.Pickler(file)
        p.dump(operator)

    os.remove(operator_path)
    try:
        os.remove(operator_path[:-4] + '_SVD.mat')
    except FileNotFoundError:
        pass
