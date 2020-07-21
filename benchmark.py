#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 27, 2019 at 14:38.

@author: a.scotto

Description:
"""

import json
import time
import numpy
import argparse
import scipy.sparse
import scipy.optimize

from core.algebra import MatrixOperator
from core.subspace import SubspaceGenerator
from core.linsolve import LinearSystem, ConjugateGradient, DeflatedConjugateGradient
from utils.utils import compute_subspace_dim, report_init, load_operator
from core.preconditioner import PreconditionerGenerator, LimitedMemoryPreconditioner

REPORTS_ROOT_PATH = 'reports/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to config file.')
parser.add_argument('operators', nargs='*', help='Path to the stored linear operator to benchmark.')
args = parser.parse_args()

with open(args.config, 'r', encoding='utf-8') as config:
    config = json.load(config)

for OPERATOR_PATH in args.operators:
    A = load_operator(OPERATOR_PATH, display=True)
    b = numpy.load(OPERATOR_PATH + '_rhs.npy')

    # Simulate recycling Krylov subspace information via random perturbation of linear system
    normalization = numpy.linalg.norm(A.mat.diagonal())
    A_ = MatrixOperator(A.mat + config['perturbation'] * normalization * scipy.sparse.eye(b.size))
    b_ = b + config['perturbation'] * normalization * numpy.random.randn(b.size, 1)

    # Recycling information
    precond_generator = PreconditionerGenerator(A)
    precond_generator_ = PreconditionerGenerator(A_)

    MAX_BUDGET = 8 * config['MEMORY_LIMIT'] * b.size

    for PRECONDITIONER in config['preconditioner']:
        M = precond_generator.get(PRECONDITIONER['name'], *PRECONDITIONER['args'], **PRECONDITIONER['kwargs'])
        linsys = LinearSystem(A, b, M)

        M_ = precond_generator_.get(PRECONDITIONER['name'], *PRECONDITIONER['args'], **PRECONDITIONER['kwargs'])
        linsys_ = LinearSystem(A_, b_, M_)

        generator = SubspaceGenerator(linsys, recycle=linsys_)

        cg_ref = ConjugateGradient(linsys, tol=config['tol'], maxiter=config['maxiter'], store_arnoldi=True)

        for SUBSPACE in config['subspaces']:
            REPORT_NAME, report_content = report_init(config, PRECONDITIONER, SUBSPACE, OPERATOR_PATH)

            report_content['reference'] = cg_ref.N
            report_content['MAX_BUDGET'] = MAX_BUDGET
            report_content['n'] = b.size

            with open(REPORTS_ROOT_PATH + REPORT_NAME, 'w') as file:
                json.dump(report_content, file, ensure_ascii=False, indent=4)

            # Compute subspace dimensions to test
            S = generator.get(SUBSPACE['name'])
            kmax_1 = scipy.optimize.fsolve(lambda c: 4*c**2 +
                                           2*A.matvec_cost +
                                           2*S.cost(c, *SUBSPACE['args'], **SUBSPACE['kwargs']) +
                                           2*S.rcost(c, *SUBSPACE['args'], **SUBSPACE['kwargs']) -
                                           MAX_BUDGET,
                                           x0=b.size)

            kmax_2 = MAX_BUDGET / (8 * b.size)

            kmax = numpy.max([kmax_1, kmax_2])

            subspace_dims = list()
            computational_cost = list()
            step = float(kmax / SUBSPACE['n_subspaces'])
            for i in range(SUBSPACE['n_subspaces']):
                subspace_dims.append(int((i + 1) * step))
                c_ = subspace_dims[-1]
                if kmax == kmax_1:
                    computational_cost.append(4*c_**2 +
                                              2*A.matvec_cost +
                                              2*S.cost(c_, *SUBSPACE['args'], **SUBSPACE['kwargs']) +
                                              2*S.rcost(c_, *SUBSPACE['args'], **SUBSPACE['kwargs']))
                else:
                    computational_cost.append(8*c_*b.size)

            print(subspace_dims)

            report_content['subspace_dims'] = subspace_dims
            report_content['computational_cost'] = computational_cost
            report_content['data'] = dict()

            for k in subspace_dims:
                data = dict(iterations=list(), time=list())
                for _ in range(SUBSPACE['n_tests']):
                    start = time.perf_counter()
                    cg = DeflatedConjugateGradient(linsys,
                                                   S.get(k, *SUBSPACE['args'], **SUBSPACE['kwargs']),
                                                   tol=config['tol'],
                                                   maxiter=config['maxiter'])
                    elapsed = time.perf_counter() - start
                    data['iterations'].append(cg.N)
                    data['time'].append(elapsed)

                report_content['data'][str(k)] = data

                with open(REPORTS_ROOT_PATH + REPORT_NAME, 'w') as file:
                    json.dump(report_content, file, ensure_ascii=False, indent=4)
