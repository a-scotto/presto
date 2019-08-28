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


def read_setup(OPERATOR_ROOT_PATH, setup_file_path):
    """
    Method to read the benchmark setup text file and return the different setup in a suitable
    format to run the benchmark.

    :param OPERATOR_ROOT_PATH: Path of folder containing the operators
    :param setup_file_path: Path of the setup text file
    """
    setups = []

    # Read the setup text file
    with open(setup_file_path, 'r') as setup:
        # Get the list of text lines
        content = setup.readlines()

        # Go through all the lines
        for line in content:
            # Skip user dedicated lines
            if line[0] == '>':
                continue

            # Turn CSV stored data into list
            test_setup = line.split(',')

            # Build the setup dictionary and append it to the setups list
            if test_setup[0] == 'all':
                # Get all the operators when 'all' is specified
                for file_name in os.listdir(OPERATOR_ROOT_PATH):
                    # Skip '__pycache__/' directory
                    if os.path.isfile(OPERATOR_ROOT_PATH + file_name):
                        setup = dict()
                        setup['operator'] = OPERATOR_ROOT_PATH + file_name
                        setup['sampling'] = test_setup[1]
                        setup['n_samples'] = int(test_setup[2])
                        setup['n_tests'] = int(test_setup[3])

                        # Get additional arguments if necessary
                        if len(test_setup) == 5:
                            setup['args'] = float(test_setup[4])
                        else:
                            setup['args'] = None

                        setups.append(setup)

            else:
                setup = dict()
                setup['operator'] = OPERATOR_ROOT_PATH + test_setup[0]
                setup['sampling'] = test_setup[1]
                setup['n_samples'] = int(test_setup[2])
                setup['n_tests'] = int(test_setup[3])

                if len(test_setup) == 5:
                    setup['args'] = float(test_setup[4])
                else:
                    setup['args'] = None

                setups.append(setup)

    return setups


def initialize_report(subspace_sizes, setup):
    """
    Method to create a report file and write the header.

    :param subspace_sizes: list of subspaces size (int) tested in the benchmark.
    :param setup: dictionary containing the setup parameters.
    """
    # Date and time fore report identification
    date_ = ''.join(time.strftime("%x").split('/'))
    time_ = ''.join(time.strftime("%X").split(':'))

    # Sampling method and eventual parameters
    param = '' if setup['args'] is None else str(setup['args'])
    sampling = setup['sampling'] + param

    # Get operator name
    operator_name = setup['operator'].split('/')[-1]

    # Set the report name from metadata above
    report_name = '_'.join([operator_name, date_, time_, sampling])

    # Load problem for metadata
    operator = load_operator(setup['operator'], display=False)
    metadata = '#'.join([str(operator['rank']),
                         str(operator['non_zeros']),
                         str(operator['conditioning']),
                         operator['source']])

    # Header writing
    with open('reports/' + report_name, 'w') as report_file:
        report_file.write('>>   REPORT OF PRECONDITIONING STRATEGIES BENCHMARK   << \n')
        report_file.write('> \n')
        report_file.write('>  SOLVER ................. conjugate gradient \n')
        report_file.write('>  NUMBER OF TESTS ........ ' + str(len(subspace_sizes)) + '\n')
        report_file.write('>  RUNS PER TEST .......... ' + str(setup['n_tests']) + '\n')
        report_file.write('>  MINIMAL SUBSPACE SIZE .. ' + str(subspace_sizes[0]) + '\n')
        report_file.write('>  MAXIMAL SUBSPACE SIZE .. ' + str(subspace_sizes[-1]) + '\n')
        report_file.write('>  PROBLEM NAME ........... ' + operator_name + '\n')
        report_file.write('> \n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        report_file.write('~metadata#' + metadata + '\n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')

    return report_name


def extract(file_path, hand_val=True):

    operator = dict()

    operator['name'] = file_path.split('/')[-1][:-4]
    operator['symmetric'] = True
    operator['def_pos'] = True

    obj = scipy.io.loadmat(file_path)['Problem']

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
        svd = scipy.io.loadmat('operators/' + operator['name'] + '_SVD')

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

        valid = input('Satisfying content ? [y/n]')

        if valid != 'y':
            raise FileExtractionError
        else:
            pass

    with open('operators/' + clean_name, 'wb') as file:
        p = pickle.Pickler(file)
        p.dump(operator)

    try:
        os.remove('operators/' + operator['name'] + '.mat')
        os.remove('operators/' + operator['name'] + '_SVD.mat')
    except FileNotFoundError:
        os.remove('operators/' + operator['name'] + '.mat')


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

