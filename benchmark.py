#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 27, 2019 at 14:38.

@author: a.scotto

Description:
"""

import numpy
import argparse

from utils.benchmarking import read_setup, set_reference_run, initialize_report, benchmark
from core.linear_operator import SelfAdjointMatrix
from core.preconditioner import DiagonalPreconditioner
from core.linear_system import ConjugateGradient, LinearSystem
from utils.utils import load_operator

OPERATOR_ROOT_PATH = 'operators/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--setup', default='setup', help='Path to benchmark setup file')
args = parser.parse_args()

# Extract the setups from benchmark setup file provided
setups = read_setup(OPERATOR_ROOT_PATH, args.setup)

for operator_path in setups.keys():
    # Load operator
    operator = load_operator(operator_path)

    # Alleviate notations by introducing parameters
    n = operator['rank']
    nnz_A = operator['non_zeros']

    # Normalize the operator to enforce 1 lying in the spectrum of the operator
    u = numpy.random.randn(operator['rank'], 1)
    gamma = float(u.T.dot(u)) / float(u.T.dot(operator['operator'].dot(u)))

    # Instantiate LinearOperator object
    lin_op = SelfAdjointMatrix(gamma * operator['operator'], def_pos=True)

    # Instantiate first-level Preconditioner object
    first_level_precond = DiagonalPreconditioner(lin_op)
    setup['first_level_precond'] = first_level_precond

    # Set the path to store the results
    RUN_STORAGE_PATH = 'reports/' + operator['name'] + '_ref'

    reference_run = set_reference_run(RUN_STORAGE_PATH, lin_op, first_level_precond)

    for setup in setups[operator_path]:
        # Add reference run to setup dictionary
        setup['reference'] = reference_run['n_iterations']

        # Define the stochastic LMP subspace sizes to be tested
        step = int(n * setup['max_ratio']) // setup['n_subspaces'] + 1
        sto_subspaces = numpy.asarray([step * (i + 1) for i in range(setup['n_subspaces'])])
        setup['sto_subspaces'] = sto_subspaces

        # Define the corresponding deterministic LMP subspace sizes
        p = numpy.array(sto_subspaces * (1 - setup['sampling_parameter']) + n, dtype=numpy.int)
        det_subspaces = (4 * nnz_A + 6 * p + sto_subspaces**2) // (8 * n) + 1
        setup['det_subspaces'] = det_subspaces

        # Initialize the report text file
        report_file_name = initialize_report(operator, setup)

        # Create the reference LinearSystem object
        lin_sys = LinearSystem(lin_op, reference_run['lhs'])

        # Benchmark the operator with setup configuration
        benchmark(lin_sys, setup)

