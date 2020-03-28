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


class TestOperator(object):

    def __init__(self, OPERATOR_FILE_PATH: str, user_check: bool = True) -> None:
        """

        :param OPERATOR_FILE_PATH:
        :param user_check: Whether to ask the user if the results match the expectations or not.
        """

        # Extract regarding the format of the file
        if not os.path.isfile(OPERATOR_FILE_PATH):
            raise FileExtractionError('No file at location: {}'.format(OPERATOR_FILE_PATH))

        if OPERATOR_FILE_PATH.endswith('.mat'):
            self.wrapper_content = _extract_mat(OPERATOR_FILE_PATH)

        elif OPERATOR_FILE_PATH.endswith('.npz'):
            self.wrapper_content = _extract_npz(OPERATOR_FILE_PATH)

        else:
            print('Format of operator not handled. Pass.')

        self.wrapper_content['rhs'] = self._set_rhs()

        # Ask for user checking if required
        if user_check:
            self.user_extraction_check()

        # Ask for file deletion
        self.ask_for_removal(OPERATOR_FILE_PATH)

    def __getitem__(self, attribute: str):
        """
        Getter of the class bridging the gap between the inner dictionary of attribute and external
        usages.

        :param attribute: Name of the attribute to get.
        """

        # Process requirements handling case of failure
        if attribute not in self.wrapper_content.keys():
            raise ValueError('TestOperator do not have "{}" attribute.'.format(attribute))

        return self.wrapper_content[attribute]

    def __setitem__(self, attribute: str, new_value: object) -> None:
        """
        Setter of the class bridging the gap between the inner dictionary of attribute and external
        usages.

        :param attribute: Name of the attribute to get.
        :param new_value: New value of the attribute to set.
        """

        # Process requirements handling case of failure
        if attribute not in self.wrapper_content.keys():
            raise ValueError('TestOperator do not have {} attribute.'.format(attribute))
        if not isinstance(new_value, type(self.wrapper_content[attribute])):
            raise ValueError('Attribute {} incorrect type.'.format(attribute))

        self.wrapper_content[attribute] = new_value

    def _set_rhs(self):
        """
        Randomly generate a right-hand side of size (n, 1).
        """

        y = numpy.random.randn(self.wrapper_content['rank'], 1)

        return self.wrapper_content['operator'] @ y

    def user_extraction_check(self) -> None:
        """
        Method to interact with the user so as to check reliability of extraction and potential
        file removals.
        """

        print()
        # Print suitably the metadata extracted
        for key, val in self.wrapper_content.items():
            if key not in ['operator', 'rhs']:
                print('{:12}: {}'.format(key.capitalize(), val))

        print()
        # Ask for user validation
        valid = ''
        while valid not in ['y', 'n']:
            valid = input('Is the extraction result satisfying? [y/n] ')

        if valid == 'n':
            raise FileExtractionError

    @staticmethod
    def ask_for_removal(OPERATOR_FILE_PATH: str) -> None:
        """
        Method to interact with the user so as to check reliability of extraction and potential
        file removals.

        :param OPERATOR_FILE_PATH: Path file to potentially remove
        """

        # Ask for user cleaning
        valid = ''
        while valid not in ['y', 'n']:
            valid = input('Do you want to remove {}? [y/n] '.format(OPERATOR_FILE_PATH))

        if valid == 'y':
            os.remove(OPERATOR_FILE_PATH)
            try:
                os.remove(OPERATOR_FILE_PATH[:-4] + '_SVD.mat')
            except FileNotFoundError:
                pass

    def __repr__(self) -> str:

        try:
            _repr = 'Problem {}: shape {} | Conditioning = {:1.2e} | Density = {:1.2e} | NNZ = {}' \
                .format(self.wrapper_content['name'],
                        self.wrapper_content['shape'],
                        self.wrapper_content['conditioning'],
                        self.wrapper_content['non_zeros'] / self.wrapper_content['shape'][0]**2,
                        self.wrapper_content['non_zeros'])
        except TypeError:
            _repr = 'Problem {}: shape {} | Density = {:1.2e} | NNZ = {}' \
                .format(self.wrapper_content['name'],
                        self.wrapper_content['shape'],
                        self.wrapper_content['non_zeros'] / self.wrapper_content['rank']**2,
                        self.wrapper_content['non_zeros'])

        return _repr


def load_operator(OPERATOR_FILE_PATH: str, display: bool = True) -> TestOperator:
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
        print(operator)

    return operator


def _extract_mat(OPERATOR_FILE_PATH: str) -> dict:
    """
    Return the content of .mat files containing matrices and meta-data as a dictionary and check
    for potential SVD decomposition .mat file.

    :param OPERATOR_FILE_PATH: Path to the file to extract from.
    """

    # Check that file do have a .mat extension
    if not OPERATOR_FILE_PATH.endswith('.mat'):
        raise FileExtensionError('Operator file should have .mat extension.')

    OPERATOR_FILE_PATH, _ = os.path.splitext(OPERATOR_FILE_PATH)
    FOLDER_PATH, FILE_NAME = os.path.split(OPERATOR_FILE_PATH)

    # Initialize the dictionary with matrix metadata
    operator = dict()

    operator['name'] = FILE_NAME.replace('_', '')
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

    return operator


def _extract_npz(OPERATOR_FILE_PATH: str) -> dict:
    """
    Return the content of .npz files containing a sparse matrix.

    :param OPERATOR_FILE_PATH: Path to the file to extract from.
    """
    # Initialize variable
    operator = dict()

    # Load de matrix
    matrix = scipy.sparse.load_npz(OPERATOR_FILE_PATH)

    # Extract available information
    operator['operator'] = matrix
    operator['non_zeros'] = matrix.size
    operator['shape'] = matrix.shape

    # Process the rank
    full_rank = input('Is matrix full-rank? [y/n]  ')
    operator['rank'] = matrix.shape[0] if full_rank == 'y' else None

    # Process the source
    source = input('What is the source of the matrix?  ')
    operator['source'] = source

    # Process the source
    name = input('What is the matrix name?  ')
    operator['name'] = name

    # Process the symmetric positive definite character
    symmetric = input('Is the matrix symmetric? [y/n]  ')
    operator['symmetric'] = True if symmetric == 'y' else None

    pos_def = input('Is the matrix positive-definite? [y/n]  ')
    operator['pos_def'] = True if pos_def == 'y' else None

    # Process the SVD decomposition
    SVD_PATH = input('Path of the SVD decomposition. Press Enter if not.  ')

    if os.path.isfile(SVD_PATH):
        operator['svd'] = numpy.load(SVD_PATH)
        operator['conditioning'] = operator['svd'][0] / operator['svd'][-1]
    else:
        operator['svd'] = None
        operator['conditioning'] = None

    return operator
