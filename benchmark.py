#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 27, 2019 at 14:38.

@author: a.scotto

Description:
"""

import json
import numpy
import scipy
import argparse

from core.linop import MatrixOperator
from core.subspace import SubspaceGenerator
from core.linsys import LinearSystem, ConjugateGradient
from utils.utils import compute_subspace_dim, report_init, load_operator
from core.preconditioner import PreconditionerGenerator, LimitedMemoryPreconditioner

REPORTS_ROOT_PATH = 'reports/'

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to config file.')
parser.add_argument('operators', nargs='*', help='Path to the stored linear operator to benchmark.')
args = parser.parse_args()

with open(args.config, 'r') as config:
    config = json.load(config)

for OPERATOR_PATH in args.operators:
    A = load_operator(OPERATOR_PATH, display=True)
    b = numpy.load(OPERATOR_PATH + '_rhs.npy')
    linsys = LinearSystem(A, b)

    precond_generator = PreconditionerGenerator(linsys.linear_op)

    # Recycling information
    recycled_linsys = linsys.perturb(config['perturbation'])
    recycled_generator = PreconditionerGenerator(recycled_linsys.linear_op)

    MAX_BUDGET = 8 * config['MEMORY_LIMIT'] * b.size

    for PRECONDITIONER in config['preconditioner']:
        M = precond_generator.get(PRECONDITIONER['name'], *PRECONDITIONER['args'], **PRECONDITIONER['kwargs'])
        M_recycled = recycled_generator.get(PRECONDITIONER['name'], *PRECONDITIONER['args'], **PRECONDITIONER['kwargs'])

        subspace_generator = SubspaceGenerator(linsys, M=M, recycling=(recycled_linsys, M_recycled))

        cg_ref = ConjugateGradient(linsys, M=M, tol=config['tol'], maxiter=config['maxiter'], store_arnoldi=True)

        for SUBSPACE in config['subspaces']:
            REPORT_NAME, report_content = report_init(config, PRECONDITIONER, SUBSPACE, OPERATOR_PATH)

            with open(REPORTS_ROOT_PATH + REPORT_NAME, 'w') as file:
                json.dump(report_content, file, ensure_ascii=False, indent=4)

            report_content['reference'] = cg_ref.N
            report_content['approximate_spectrum'] = list(scipy.linalg.eigvalsh(cg_ref.H[:-1, :]))
            report_content['subspace_format'] = subspace_generator.output_format[SUBSPACE['name']]
            report_content['MAX_BUDGET'] = MAX_BUDGET
            report_content['n'] = b.size

            subspace_dims, computational_cost = compute_subspace_dim(MAX_BUDGET,
                                                                     SUBSPACE['n_subspaces'],
                                                                     linsys.linear_op,
                                                                     report_content['subspace_format'])

            report_content['subspace_dims'] = subspace_dims
            report_content['computational_cost'] = computational_cost
            report_content['data'] = dict()

            for k in subspace_dims:
                data = list()
                for _ in range(SUBSPACE['n_tests']):
                    S = subspace_generator.get(SUBSPACE['name'], k, *SUBSPACE['args'], **SUBSPACE['kwargs'])
                    H = LimitedMemoryPreconditioner(linsys.linear_op, S, M)
                    cg = ConjugateGradient(linsys, M=H, tol=config['tol'], maxiter=config['maxiter'])
                    data.append(cg.N)

                report_content['data'][str(k)] = data

                with open(REPORTS_ROOT_PATH + REPORT_NAME, 'w') as file:
                    json.dump(report_content, file, ensure_ascii=False, indent=4)
