#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 14:04.

@author: a.scotto

Description:
"""

import os
import numpy
import pickle
import argparse
import scipy.io
import scipy.sparse

from core.algebra import MatrixOperator

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('files',
                    nargs='*',
                    help='Paths to files containing stored matrices.')

args = parser.parse_args()

for FILE_PATH in args.files:

    # Skip .mat files containing SVD decomposition
    if not FILE_PATH.endswith('.mat'):
        continue

    mat = scipy.io.loadmat(FILE_PATH)['Problem']

    while len(mat) == 1:
        mat = mat[0]

    for item in mat:
        if scipy.sparse.isspmatrix(item):
            mat = MatrixOperator(item)
            break

    name, _ = os.path.basename(FILE_PATH).split('.')

    with open('problems/' + name, 'wb') as file:
        p = pickle.Pickler(file)
        p.dump(mat)

    rhs = input('Generate random right-hand side? [y/n] ... ')

    if rhs == 'y':
        X = numpy.random.randn(mat.shape[0])
        diag = scipy.sparse.diags(mat.mat.diagonal()**0.5)
        sigma = 1 / mat.shape[0]**0.5 * numpy.linalg.norm(mat.mat.diagonal())
        rhs = numpy.asarray([b for b in [mat @ X, diag @ X, float(sigma) * X]])
        numpy.save('problems/' + name + '_rhs', rhs)
