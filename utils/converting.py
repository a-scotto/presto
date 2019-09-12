#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 13:48.

@author: a.scotto

Description:
"""

import os
import numpy
import pickle
import scipy.io
import scipy.linalg
import scipy.sparse


class FileExtractionError(Exception):
    """
    Exception raised when the extraction of a MATLAB file fails.
    """


class FileExtensionError(Exception):
    """
    Exception raised when the extension of a specified file do not match requirements.
    """


def load_operator(OPERATOR_FILE_PATH: str, display: bool = True) -> dict:
    """
    Load the binary operator file located at the specified path and return it as dictionary.

    :param OPERATOR_FILE_PATH: Path to the .opr operator file.
    :param display: Whether to display operator's characteristics or not.
    """

    # Check that file do have a .opr extension
    if not OPERATOR_FILE_PATH.endswith('.opr'):
        raise FileExtensionError('Operator file should have .opr extension.')

    # Open the operator binary file and load the content
    with open(OPERATOR_FILE_PATH, 'rb') as file:
        p = pickle.Unpickler(file)
        operator = p.load()

    # Display the operator characteristics if required
    if display:
        print('Problem {}: shape {} | Conditioning = {:1.2e} | Density = {:1.2e} | NNZ = {}'
              .format(operator['name'],
                      operator['shape'],
                      operator['conditioning'],
                      operator['non_zeros'] / operator['shape'][0]**2,
                      operator['non_zeros']))

    return operator


def extract(OPERATOR_FILE_PATH: str, user_check: bool = True) -> dict:
    """
    Return the content of .mat files containing matrices and meta-data as a dictionary and check
    for potential SVD decomposition .mat file.

    :param OPERATOR_FILE_PATH: Path to the file to extract from.
    :param user_check: Whether to ask the user if the results match the expectations or not.
    """

    # Check that file do have a .mat extension
    if not OPERATOR_FILE_PATH.endswith('.mat'):
        raise FileExtensionError('Operator file should have .mat extension.')

    OPERATOR_FILE_PATH, _ = os.path.splitext(OPERATOR_FILE_PATH)
    FOLDER_PATH, FILE_NAME = os.path.split(OPERATOR_FILE_PATH)

    # Initialize the dictionary with matrix metadata
    operator = dict()

    operator['name'] = FILE_NAME.replace('_', '') + '.opr'
    operator['symmetric'] = True
    operator['def_pos'] = True

    # Load the .mat file
    obj = scipy.io.loadmat(OPERATOR_FILE_PATH)['Problem']

    # Skip the useless containers
    while len(obj) == 1:
        obj = obj[0]

    # Search for the Sparse Matrix object stored
    for entree in obj:
        if isinstance(entree, numpy.ndarray) and entree.size == 1:
            entree = entree[0]

        if scipy.sparse.isspmatrix(entree) and 'operator' not in operator.keys():
            operator['operator'] = entree
            operator['non_zeros'] = entree.size
            operator['shape'] = entree.shape
            operator['rank'] = entree.shape[0]

        # Get the problem source
        elif isinstance(entree, str) and 'problem' in entree.lower():
            operator['source'] = entree.capitalize()

    # Try to access the SVD decomposition file if available
    SVD_FILE_PATH = os.path.join(FOLDER_PATH, FILE_NAME + '_SVD')

    try:
        svd = scipy.io.loadmat(SVD_FILE_PATH)

        # Skip the useless containers
        while len(svd) == 1:
            svd = svd[0]

        svd = svd['S']

        while len(svd) == 1:
            svd = svd[0]

        for entree in svd:
            if entree.size == operator['rank']:
                operator['svd'] = entree
                operator['conditioning'] = float(entree[0] / entree[-1])

    # Handle the case where the SVD decomposition is not available
    except FileNotFoundError:
        operator['svd'] = None
        operator['conditioning'] = None

    # Let the user manually check the information extracted
    if user_check:
        for key, val in operator.items():
            print('{}: {}'.format(key, val))

        valid = ''
        while valid not in ['y', 'n']:
            valid = input('Is the extraction result satisfying? [y/n] ')

        if valid == 'n':
            raise FileExtractionError

    return operator
