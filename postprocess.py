#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 20, 2019 at 09:09.

@author: a.scotto

Description:
"""

import os
import numpy
import argparse

from matplotlib import pyplot

COLORS = ['tab:blue',
          'tab:orange',
          'tab:green',
          'tab:red',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'tab:gray',
          'tab:olive',
          'tab:cyan']

MARKER = ['', 'o', 's', '^']

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-r',
                    nargs='*',
                    required=True,
                    dest='reports',
                    help='Paths of reports text files to post-process.')
parser.add_argument('-c',
                    required=True,
                    dest='comparison',
                    choices=['pb', 'spl'],
                    help='Comparison criteria, either problems or sampling methods.')
args = parser.parse_args()
comparison = args.comparison

reports_sets = dict()

# Pre-processing the files path provided regarding the comparison choice
for REPORT_PATH in args.reports:
    _, report_name = os.path.split(REPORT_PATH)

    problem_name = report_name.split('_')[0]
    sampling = report_name.split('_')[-1]

    # Sort the reports according to the comparison criterion provided
    if comparison == 'spl':
        if problem_name not in reports_sets.keys():
            reports_sets[problem_name] = [REPORT_PATH]
        else:
            reports_sets[problem_name].append(REPORT_PATH)
    else:
        if sampling not in reports_sets.keys():
            reports_sets[sampling] = [REPORT_PATH]
        else:
            reports_sets[sampling].append(REPORT_PATH)

# Go through all the reports
for key, reports_paths in reports_sets.items():
    pyplot.figure()
    pyplot.grid()

    # Set corresponding title regarding the comparison criterion
    if comparison == 'spl':
        plot_title = 'Sampling strategies results on ' + key + ' problem'
    else:
        plot_title = 'Operators preconditioned by ' + key + ' sampled subspaces.'

    # Browse through the reports file names in the report set
    for k, REPORT_PATH in enumerate(reports_paths):
        _, report_name = os.path.split(REPORT_PATH)

        if comparison == 'spl':
            label = report_name.split('_')[-1]
        else:
            label = report_name.split('_')[0]

        subspace_sizes, means, stdevs, minima, maxima = [], [], [], [], []

        # Read and store the data contained in reports files
        with open(REPORT_PATH, 'r') as report:
            # Get the list of text lines
            content = report.readlines()

            for data in content:
                # Skip report header
                if data[0] == '>':
                    pass

                # Read metadata
                elif data[0] == '~':
                    metadata = data.split('#')[1:]
                    size = int(metadata[0])
                    nnz = int(metadata[1])
                    cond = float(metadata[2])
                    source = metadata[3]

                # Read data
                else:
                    p, data = data.split('_')
                    perf_ratios = data.split(',')
                    perf_ratios = list(map(lambda x: float(x), perf_ratios))

                    subspace_sizes.append(int(p) / size)
                    means.append(numpy.mean(perf_ratios))
                    stdevs.append(numpy.std(perf_ratios))
                    minima.append(min(perf_ratios))
                    maxima.append(max(perf_ratios))

        lower = [means[i] - stdevs[i] for i in range(len(means))]
        upper = [means[i] + stdevs[i] for i in range(len(means))]

        # Add plot to the current figure
        pyplot.plot(subspace_sizes,
                    means,
                    linestyle='-',
                    mec='k',
                    ms=5,
                    marker=MARKER[(k // len(COLORS)) % len(MARKER)],
                    color=COLORS[k % len(COLORS)],
                    label=label)

        pyplot.fill_between(subspace_sizes, lower, upper, color=COLORS[k % len(COLORS)], alpha=0.2)
        pyplot.plot(subspace_sizes, minima, color=COLORS[k % len(COLORS)], marker='+', lw=0)

    # Add the legend, title, and axis titles
    pyplot.legend()
    pyplot.title(plot_title)
    pyplot.xlabel('Ratios of subspace size and problem size k / n')
    pyplot.ylabel('Ratios of iterations between CG and PCG')

    pyplot.xlim(subspace_sizes[0], subspace_sizes[-1])
    pyplot.ylim(bottom=0.)

pyplot.show()
