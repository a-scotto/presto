#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on September 06, 2019 at 09:43.

@author: a.scotto

Description:
"""

import os
import tqdm
import numpy
import scipy.sparse

from core.linear_system import ConjugateGradient, LinearSystem
from core.random_subspace import RandomSubspaceFactory
from core.preconditioner import LimitedMemoryPreconditioner, DiagonalPreconditioner
from core.linear_operator import SelfAdjointMatrix


def read_setup(OPERATOR_ROOT_PATH, setup_file_path):
    """
    Method to read the benchmark setup text file and return the different setup in a suitable
    format to run the benchmark.

    :param OPERATOR_ROOT_PATH: Path of folder containing the operators
    :param setup_file_path: Path of the setup text file
    """

    # Get the list of all available operators in OPERATOR_ROOT_PATH folder
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
            if line.startswith('>'):
                continue

            # Turn spaced separated data into list
            operators, n_subspaces, max_ratio, n_tests, sampling = line.split(' ')

            # Get the sampling method and optionally arguments
            sampling_method, sampling_parameter = sampling.split(',')

            setup = dict(n_subspaces=int(n_subspaces),
                         max_ratio=float(max_ratio),
                         n_tests=int(n_tests),
                         sampling_method=sampling_method,
                         sampling_parameter=float(sampling_parameter))

            # Either add all operators or the desired one to the setups dictionary
            if operators == '*':
                for operator in operators_list:
                    OPERATOR_PATH = os.path.join(OPERATOR_ROOT_PATH, operator)

                    # Add the operator path if already existing, else create the list
                    try:
                        setups[OPERATOR_PATH].append(setup)
                    except KeyError:
                        setups[OPERATOR_PATH] = [setup]
            else:
                OPERATOR_PATH = os.path.join(OPERATOR_ROOT_PATH, operators)

                if operators not in operators_list:
                    raise FileNotFoundError('Not such operator file {}.'.format(OPERATOR_PATH))

                # Add the operator path if already existing, else create the list
                try:
                    setups[OPERATOR_PATH].append(setup)
                except KeyError:
                    setups[OPERATOR_PATH] = [setup]

    return setups


def get_reference_run(operator, n_runs=50):

    RUN_STORAGE_PATH = 'reports/' + operator['name'] + '_ref'

    # Normalize the operator to enforce 1 lying in the spectrum of the operator
    u = numpy.random.randn(operator['rank'], 1)
    gamma = float(u.T.dot(u)) / float(u.T.dot(operator['operator'].dot(u)))

    # Instantiate LinearOperator object
    A = SelfAdjointMatrix(gamma * operator['operator'], def_pos=True)

    # Instantiate first-level Preconditioner object
    D = DiagonalPreconditioner(A)

    # Either load left-hand side from existing file or compute it
    try:
        reference = numpy.loadtxt(RUN_STORAGE_PATH)
        print('Reference loaded.')
        n_iterations = int(reference[0])
        lhs = reference[1:].reshape(-1, 1)

    except OSError:
        # Set a CG run of reference
        lhs = list()
        n_iterations = list()

        # Run the CG with several left-hand sides
        for i in range(n_runs):
            # Create random left-hand side
            y = numpy.random.randn(operator['rank'], 1)
            lin_sys = LinearSystem(A, A.dot(y))

            # Run the Conjugate Gradient
            cg = ConjugateGradient(lin_sys, x_0=None, M=D).run()

            # Store results
            lhs.append(lin_sys.lhs)
            n_iterations.append(cg['n_it'])

        # Get the left-hand side the closest to the average run
        average_run = int(numpy.argmin(numpy.asarray(n_iterations) - numpy.mean(n_iterations)))
        lhs = lhs[average_run]
        n_iterations = n_iterations[average_run]

        # Store left-hand side and iterations number in a single stacked array
        reference = numpy.vstack([n_iterations, lhs])
        numpy.savetxt(RUN_STORAGE_PATH, reference)
        print('Reference created.')

    lin_sys = LinearSystem(A, lhs)
    cg = ConjugateGradient(lin_sys, M=D, buffer=operator['rank'], tol=1e-10, arnoldi=True).run()
    cg['n_it'] = n_iterations

    return cg


def benchmark(setup, operator):

    n = operator['rank']

    # Test all the subspace sizes
    for k in tqdm.tqdm(setup['subspaces']):
        factory = RandomSubspaceFactory((n, k))

        # Run of the PCG with deterministic LMP
        p = int(k + (n - k) * setup['sampling_parameter'])


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
