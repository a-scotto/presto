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
from core.projection_subspace import DeterministicSubspaceFactory, RandomSubspaceFactory
from utils.benchmarking import read_setup, compute_subspace_sizes, benchmark

OPERATOR_ROOT_PATH = 'operators/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('setup', default='setup', help='Path to benchmark setup file')
args = parser.parse_args()


for setup in read_setup(args.setup):

    for operator_path in setup['operators']:

        operator = load_operator(operator_path, display=True)

        # Scale the operator to enforce 1 lying in the spectrum of the operator
        u = numpy.random.randn(operator['rank'], 1)
        gamma = float(u.T.dot(u)) / float(u.T.dot(operator['operator'].dot(u)))
        operator['operator'] = gamma * operator['operator']

        # Compute subspace sizes
        if setup['subspace']['name'] in DeterministicSubspaceFactory.subspace_type.keys():
            _type = DeterministicSubspaceFactory.subspace_type[setup['subspace']['name']]

            subspaces = compute_subspace_sizes(setup['n_subspaces'],
                                               operator['operator'],
                                               subspace_type=_type)

        elif setup['subspace']['name'] in RandomSubspaceFactory.subspace_type.keys():
            _type = RandomSubspaceFactory.subspace_type[setup['subspace']['name']]

            subspaces = compute_subspace_sizes(setup['n_subspaces'],
                                               operator['operator'],
                                               subspace_type='sparse')

        else:
            raise ValueError('Subspace type unrecognized.')

        # Process benchmark
        benchmark = (operator, setup, subspaces)
