#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 14:04.

@author: a.scotto

Description:
"""

import pickle
import fnmatch
import argparse

from utils.operator import TestOperator


# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('files',
                    nargs='*',
                    help='Paths to files containing stored matrices.')

args = parser.parse_args()

# Filter the content of the folder to select only the .mat files
operators_files = fnmatch.filter(args.files, '*.*')

for OPERATOR_PATH in operators_files:

    # Skip .mat files containing SVD decomposition
    if OPERATOR_PATH.endswith('_SVD.mat'):
        continue

    if not OPERATOR_PATH.endswith('.mat') and not OPERATOR_PATH.endswith('.npz'):
        continue

    # Extract content
    print('Extracting from {}... '.format(OPERATOR_PATH))
    operator = TestOperator(OPERATOR_PATH)
    print('Done.')

    name = operator['name']

    with open('operators/' + str(name) + '.opr', 'wb') as file:
        p = pickle.Pickler(file)
        p.dump(operator)
