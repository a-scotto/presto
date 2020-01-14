#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 27, 2019 at 14:38.

@author: a.scotto

Description:
"""

import numpy
import argparse

from utils.operator import load_operator
from core.projection_subspace import KrylovSubspaceFactory, RandomSubspaceFactory
from utils.benchmarking import read_setup, compute_subspace_sizes, initialize_report, \
                               set_reference_run, benchmark

OPERATOR_ROOT_PATH = 'operators/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--setup', default='setup', help='Path to benchmark setup file')
args = parser.parse_args()


for setup in read_setup(args.setup):

    for operator_path in setup['operators']:

        operator = load_operator(operator_path, display=True)

        # Scale the operator to enforce 1 lying in the spectrum of the operator
        u = numpy.random.randn(operator.get('rank'), 1)
        gamma = float(u.T.dot(u)) / float(u.T.dot(operator.get('operator').dot(u)))
        operator.set('operator', gamma * operator.get('operator'))

        # Compute subspace sizes
        if setup['subspace']['name'] in KrylovSubspaceFactory.krylov_type.keys():
            _type = KrylovSubspaceFactory.krylov_type[setup['subspace']['name']]

            subspaces = compute_subspace_sizes(setup['n_subspaces'],
                                               operator.get('operator'),
                                               subspace_type=_type)

        elif setup['subspace']['name'] in RandomSubspaceFactory.sampling_type.keys():
            _type = RandomSubspaceFactory.sampling_type[setup['subspace']['name']]

            subspaces = compute_subspace_sizes(setup['n_subspaces'],
                                               operator.get('operator'),
                                               subspace_type='sparse')

        else:
            raise ValueError('Subspace type unrecognized.')

        # Load or set reference run
        reference = set_reference_run(operator, setup['first_level_precond'])

        # Initialize report text file
        setup['report_name'] = initialize_report(operator, setup, reference)

        # Process benchmark
        benchmark(setup, subspaces, reference)
