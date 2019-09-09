#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 13:48.

@author: a.scotto

Description:
"""

import os
import time
import numpy
import random
import scipy.io
import scipy.linalg
import scipy.sparse
import pickle


class FileExtractionError(Exception):
    """
    Exception raised when the extraction of a MATLAB file occurs.
    """


def load_operator(file_path, display=True):
    """
    Method to load operator as dictionary from files in operators/ folder.

    :param file_path: Path of the operator to load
    :param display: Boolean to whether display or not the problem metadata
    """

    # Open the file as binary
    with open(file_path, 'rb') as file:
        p = pickle.Unpickler(file)
        operator = p.load()

    # Display if required
    if display:
        print('Problem {}: shape {} | Conditioning = {:1.2e} | Density = {:1.2e} | NNZ = {}'
              .format(operator['name'],
                      operator['shape'],
                      operator['conditioning'],
                      operator['non_zeros'] / operator['shape'][0]**2,
                      operator['non_zeros']))

    return operator


def extract(OPERATOR_FILE_PATH, hand_val=True):
    OPERATOR_FILE_PATH, _ = os.path.splitext(OPERATOR_FILE_PATH)
    FOLDER_PATH, FILE_NAME = os.path.split(OPERATOR_FILE_PATH)

    operator = dict()

    operator['name'] = FILE_NAME
    operator['symmetric'] = True
    operator['def_pos'] = True

    obj = scipy.io.loadmat(OPERATOR_FILE_PATH)['Problem']

    while len(obj) == 1:
        obj = obj[0]

    for entree in obj:

        if isinstance(entree, numpy.ndarray) and entree.size == 1:
            entree = entree[0]

        if scipy.sparse.isspmatrix(entree) and 'operator' not in operator.keys():
            operator['operator'] = entree
            operator['non_zeros'] = entree.size
            operator['shape'] = entree.shape
            operator['rank'] = entree.shape[0]

        elif isinstance(entree, str) and 'problem' in entree.lower():
            operator['source'] = entree.capitalize()

    try:
        SVD_FILE_PATH = os.path.join(FOLDER_PATH, FILE_NAME + '_SVD')
        svd = scipy.io.loadmat(SVD_FILE_PATH)

        while len(svd) == 1:
            svd = svd[0]

        svd = svd['S']

        while len(svd) == 1:
            svd = svd[0]

        for entree in svd:
            if entree.size == operator['rank']:
                operator['svd'] = entree
                operator['conditioning'] = float(entree[0] / entree[-1])

    except FileNotFoundError:
        operator['svd'] = None
        operator['conditioning'] = None

    clean_name = ''.join(operator['name'].split('_'))

    if hand_val:
        for key, val in operator.items():
            print('{}: {}'.format(key, val))

        valid = input('Satisfying content ? [y/n] ')

        if valid != 'y':
            raise FileExtractionError
        else:
            pass

    with open('operators/' + clean_name, 'wb') as file:
        p = pickle.Pickler(file)
        p.dump(operator)

    try:
        os.remove(OPERATOR_FILE_PATH + '.mat')
        os.remove(SVD_FILE_PATH + '.mat')
    except FileNotFoundError:
        os.remove(OPERATOR_FILE_PATH + '.mat')


def convert_to_col(X):
    """
    Convert numpy.ndarray to column stack format, i.e. of shape (m, n) with m > n.

    :param X: numpy.ndarray with arbitrary shape
    :return: numpy.ndarray with shape (m, n) with m > n.
    """

    if len(X.shape) == 1:
        X = numpy.atleast_2d(X).T

    if len(X.shape) != 2:
        raise ValueError('Impossible to convert to columns numpy.ndarray of dimension =/= 2.')

    if X.shape[1] > X.shape[0]:
        X = X.T

    return X


def split_lhs(lhs, n_slice, randomize=True, return_sparse=True):
    """

    :param lhs:
    :param n_slice:
    :param randomize:
    :param return_sparse:
    :return:
    """
    lhs = convert_to_col(lhs)
    slice_size = lhs.size // n_slice + 1
    index = [k for k in range(lhs.size)]

    if randomize:
        random.shuffle(index)

    lhs_slices = []

    for k in range(n_slice):
        if k < n_slice - 1:
            ind = index[k * slice_size:(k + 1) * slice_size]
        else:
            ind = index[k * slice_size:]

        lhs_s = numpy.zeros_like(lhs)
        lhs_s[ind] = lhs[ind]
        lhs_slices.append(lhs_s)

    lhs_slices = numpy.hstack(lhs_slices)

    if return_sparse:
        lhs_slices = scipy.sparse.csr_matrix(lhs_slices)

    return lhs_slices

