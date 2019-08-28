#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 27, 2019 at 14:38.

@author: a.scotto

Description:
"""

import tqdm
import numpy
import argparse

from core.linear_operator import SelfAdjointMatrix
from core.random_subspace import RandomSubspaceFactory
from core.linear_system import ConjugateGradient, LinearSystem
from tools.utils import initialize_report, load_operator, read_setup
from core.preconditioner import LimitedMemoryPreconditioner, DiagonalPreconditioner

OPERATOR_ROOT_PATH = 'operators/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--setup', default='setup', help='Path to benchmark setup file')
args = parser.parse_args()

# Extract the setups from benchmark setup file provided
setups = read_setup(OPERATOR_ROOT_PATH, args.setup)

for setup in setups:
    # Load operator
    operator = load_operator(setup['operator'])

    # Normalize to enforce that 1 lie within the spectrum of A
    u = numpy.random.randn(operator['rank'], 1)
    gamma = float(u.T.dot(u)) / float(u.T.dot(operator['operator'].dot(u)))

    # Instantiate operator and first-level preconditioning
    A = SelfAdjointMatrix(gamma * operator['operator'], def_pos=True)
    D = DiagonalPreconditioner(A)

    # Define the list of subspaces size to be tested regarding the setup parameters
    step = (operator['rank'] // 10) // setup['n_samples'] + 1
    subspace_sizes = [step * (i + 1) for i in range(setup['n_samples'])]

    # Initialize the report text file
    report_file_name = initialize_report(subspace_sizes, setup)

    # Test all the subspace sizes
    for p in tqdm.tqdm(subspace_sizes):
        REPORT_LINE = str(p) + '_'
        factory = RandomSubspaceFactory((operator['rank'], p))

        # Run the number of tests specified
        for s in range(setup['n_tests']):
            # Left-hand side and initial guess random generation
            b = gamma * numpy.random.randn(A.shape[0], 1)
            x_0 = numpy.random.randn(A.shape[0], 1)

            # Create the linear system object
            lin_sys = LinearSystem(A, b)

            # Run the Conjugate Gradient to obtain the reference run
            cg = ConjugateGradient(lin_sys, x_0=x_0, M=D).run()

            # Generate the subspace from the factory
            subspace = factory.generate(setup['sampling'], setup['args'])

            # Build the LMP from the subspace generated
            lmp = LimitedMemoryPreconditioner(A, subspace, M=D)

            # Run the Preconditioned Conjugate Gradient
            pcg = ConjugateGradient(lin_sys, x_0=x_0, M=lmp).run()

            # Add result to the report data line
            REPORT_LINE += str(pcg['n_it'] / cg['n_it'])[:6] + ','

        # Remove last coma from report line
        REPORT_LINE = REPORT_LINE[:-1]

        # Write data line in report file
        with open('reports/' + report_file_name, 'a') as report_file:
            report_file.write(REPORT_LINE + '\n')

    print()
