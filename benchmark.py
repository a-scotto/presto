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
from core.linear_operator import SelfAdjointMatrix
from core.preconditioner import DiagonalPreconditioner, SymmetricSuccessiveOverRelaxation
from utils.benchmarking import read_setup, set_reference_run, initialize_report, benchmark

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

    # Set the path to store the results
    REFERENCE_RUN_PATH = 'runs/' + operator['name'] + '_' + first_level_precond.name + '.ref'

    # Set/get the reference run to compare with.
    reference_run = set_reference_run(REFERENCE_RUN_PATH, lin_op, first_level_precond)

    # Get the number of iterations corresponding to the tolerance tested
    tolerance = 1e-5
    n_iterations = 0
    residues = reference_run.output['residues']
    for i in range(reference_run.output['n_iterations']):
        if residues[i] / residues[0] < tolerance:
            n_iterations = i
            break

    for setup in setups[operator_path]:
        # Add reference iterations number and first-level preconditioner to setup dictionary
        setup['reference'] = n_iterations
        setup['first_lvl_preconditioner'] = first_level_precond.name

        # Define the stochastic LMP subspace sizes to be tested
        step = int(n * setup['max_ratio']) // setup['n_subspaces'] + 1
        sto_subspaces = numpy.asarray([step * (i + 1) for i in range(setup['n_subspaces'])])
        setup['sto_subspaces'] = sto_subspaces.astype(int)

        # Define the corresponding deterministic LMP subspace sizes
        p = numpy.array(sto_subspaces * (1 - setup['sampling_parameter']) + n)
        det_subspaces = (2*lin_op.apply_cost + 6*p + 4*sto_subspaces**2) // (8*n) + 1
        setup['det_subspaces'] = det_subspaces.astype(int)

        # Initialize the report
        setup['report_path'] = initialize_report(operator, setup)

        # Benchmark the operator with setup configuration
        benchmark(reference_run, setup)
