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

    operators_list = os.listdir(OPERATOR_ROOT_PATH)
    operators_list.remove('__pycache__')

    setups = dict()

    # Read the setup text file
    with open(setup_file_path, 'r') as setup_text_file:
        # Get the list of text lines
        content = setup_text_file.readlines()

        # Go through all the lines
        for line in content:
            # Skip user dedicated header
            if line[0] == '>':
                continue

            # Turn spaced separated data into list
            operator, n_subspaces, ratio_max, n_tests, sampling_parameters = line.split(' ')

            # Get the sampling method and optionally arguments
            try:
                sampling_method, args = sampling_parameters.split(',')
            except ValueError:
                sampling_method = sampling_parameters
                args = None

            setup = dict(n_subspaces=int(n_subspaces),
                         ratio_max=float(ratio_max),
                         n_tests=int(n_tests),
                         sampling=sampling_method,
                         args=float(args))

            # Build the setup dictionary and append it to the setups list
            if operator == '*':
                for operator in operators_list:
                    try:
                        setups[OPERATOR_ROOT_PATH + operator].append(setup)
                    except KeyError:
                        setups[OPERATOR_ROOT_PATH + operator] = [setup]
            else:
                if operator not in operators_list:
                    raise ValueError('Operator {} not in {}.'.format(operator, OPERATOR_ROOT_PATH))

                try:
                    setups[OPERATOR_ROOT_PATH + operator].append(setup)
                except KeyError:
                    setups[OPERATOR_ROOT_PATH + operator] = [setup]

    return setups


def initialize_report(subspace_sizes, operator_path, setup):
    """
    Method to create a report file and write the header.

    :param subspace_sizes: list of subspaces size (int) tested in the benchmark.
    :param operator_path: path of the operator to be tested.
    :param setup: dictionary containing the setup parameters.
    """
    # Date and time fore report identification
    date_ = ''.join(time.strftime("%x").split('/'))
    time_ = ''.join(time.strftime("%X").split(':'))

    # Sampling method and eventual parameters
    param = '' if setup['args'] is None else str(setup['args'])
    sampling = setup['sampling'] + param

    # Get operator name
    operator_name = os.path.basename(operator_path)

    # Set the report name from metadata above
    report_name = '_'.join([operator_name, date_, time_, sampling])

    # Load problem for metadata
    operator = load_operator('operators/' + operator_name, display=False)

    operator_metadata = '#'.join([str(operator['rank']),
                                  str(operator['non_zeros']),
                                  str(operator['conditioning']),
                                  operator['source']])

    benchmark_metadata = '#'.join([str(setup['reference']),
                                   str(param)])

    # Header writing
    with open('reports/' + report_name, 'w') as report_file:
        report_file.write('>>   REPORT OF PRECONDITIONING STRATEGIES BENCHMARK   << \n')
        report_file.write('> \n')
        report_file.write('>  SOLVER ................. Conjugate Gradient \n')
        report_file.write('>  NUMBER OF TESTS ........ ' + str(len(subspace_sizes)) + '\n')
        report_file.write('>  RUNS PER TEST .......... ' + str(setup['n_tests']) + '\n')
        report_file.write('>  MINIMAL SUBSPACE SIZE .. ' + str(subspace_sizes[0]) + '\n')
        report_file.write('>  MAXIMAL SUBSPACE SIZE .. ' + str(subspace_sizes[-1]) + '\n')
        report_file.write('>  PROBLEM NAME ........... ' + operator_name + '\n')
        report_file.write('>  REFERENCE RUN ........... ' + str(setup['reference']) + '\n')
        report_file.write('> \n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        report_file.write('~operator_metadata#' + operator_metadata + '\n')
        report_file.write('~benchmark_metadata#' + benchmark_metadata + '\n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')

    return report_name


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

