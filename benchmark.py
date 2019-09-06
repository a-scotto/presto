#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 27, 2019 at 14:38.

@author: a.scotto

Description:
"""

import numpy
import argparse

from utils.benchmarking import read_setup, get_reference_run
from core.linear_operator import SelfAdjointMatrix
from core.preconditioner import DiagonalPreconditioner
from core.linear_system import ConjugateGradient, LinearSystem
from utils.utils import initialize_report, load_operator

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

    reference_run = get_reference_run(operator)

    for setup in setups[operator_path]:
        # Add reference run to setup dictionary
        setup['reference'] = reference_run['n_it']

        # Define the list of subspaces size to be tested regarding the setup parameters
        step = int(n * setup['max_ratio']) // setup['n_subspaces'] + 1
        sto_subspaces = numpy.asarray([step * (i + 1) for i in range(setup['n_subspaces'])])

        p = numpy.array(setup['subspaces'] * (1 - setup['sampling_parameter']) + n, dtype=numpy.int)
        det_subspaces = (4 * nnz_A + 6 * p + setup['subspaces']**2) // (8 * n) + 1

        # Initialize the report text file
        report_file_name = initialize_report(operator_path, setup)

        # Benchmark the operator with setup configuration

