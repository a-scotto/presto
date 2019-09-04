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
import scipy.sparse

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

for operator_path in setups.keys():
    # Load operator
    operator = load_operator(operator_path)
    n = operator['rank']

    # Normalize to enforce that 1 lie within the spectrum of A
    u = numpy.random.randn(n, 1)
    gamma = float(u.T.dot(u)) / float(u.T.dot(operator['operator'].dot(u)))

    # Instantiate operator and first-level preconditioning
    A = SelfAdjointMatrix(gamma * operator['operator'], def_pos=True)
    D = DiagonalPreconditioner(A)

    # Set a CG run of reference
    lhs = list()
    n_iterations = list()

    # Run the CG with several left-hand sides
    for i in range(50):
        lhs.append(A.dot(numpy.random.randn(A.shape[0], 1)))
        lin_sys = LinearSystem(A, lhs[-1])

        cg = ConjugateGradient(lin_sys, x_0=None, M=D).run()
        n_iterations.append(cg['n_it'])

    # Get the LHS close to the average run
    average_run = int(numpy.argmin(numpy.asarray(n_iterations) - numpy.mean(n_iterations)))
    lin_sys = LinearSystem(A, lhs[average_run])
    n_iterations_ref = n_iterations[average_run]
    cg = ConjugateGradient(lin_sys, x_0=None, M=D, buffer=n, tol=1e-8).run()

    for setup in setups[operator_path]:
        # Define the list of subspaces size to be tested regarding the setup parameters
        step = int(n * setup['ratio_max']) // setup['n_subspaces'] + 1
        subspace_sizes = [step * (i + 1) for i in range(setup['n_subspaces'])]
        setup['reference'] = n_iterations_ref

        # Initialize the report text file
        report_file_name = initialize_report(subspace_sizes, operator_path, setup)

        # Test all the subspace sizes
        for k in tqdm.tqdm(subspace_sizes):
            factory = RandomSubspaceFactory((n, k))

            # Run of the PCG with deterministic LMP
            p = int(k + (n - k) * setup['args'])
            k_0 = (4*operator['non_zeros'] + 6*p + p**2) // (8 * n) + 1
            subspace = numpy.hstack(cg['p'][:k_0])
            lmp_det = LimitedMemoryPreconditioner(A, subspace, M=D)
            pcg = ConjugateGradient(lin_sys, x_0=None, M=lmp_det).run()

            deterministic_report = str(k_0) + '_' + str(pcg['n_it'] / n_iterations_ref)[:4]

            REPORT_LINE = deterministic_report + '#' + str(k) + '_'

            best = numpy.Inf
            worse = 0

            # Run the number of tests specified
            for i in range(setup['n_tests']):
                # Generate the subspace from the factory
                subspace = factory.generate(setup['sampling'], setup['args'])

                # Build the LMP from the subspace generated
                lmp = LimitedMemoryPreconditioner(A, subspace, M=D)

                # Run the Preconditioned Conjugate Gradient
                pcg = ConjugateGradient(lin_sys, x_0=None, M=lmp).run()
                pcg_score = pcg['n_it'] / n_iterations_ref

                # Add result to the report data line
                REPORT_LINE += str(pcg_score)[:6] + ','

                if k == subspace_sizes[-1]:
                    if pcg_score > worse:
                        scipy.sparse.save_npz('reports/' + report_file_name + '_worse_' + str(k),
                                              subspace)
                        worse = pcg_score

                    elif pcg_score < best:
                        scipy.sparse.save_npz('reports/' + report_file_name + '_best_' + str(k),
                                              subspace)
                        best = pcg_score

            # Remove last coma from report line
            REPORT_LINE = REPORT_LINE[:-1]

            # Write data line in report file
            with open('reports/' + report_file_name, 'a') as report_file:
                report_file.write(REPORT_LINE + '\n')

        print()
