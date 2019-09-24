#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 20, 2019 at 09:09.

@author: a.scotto

Description:
"""

import os
import argparse

from matplotlib import pyplot, rc
from utils.postprocessing import read_report

rc('text', usetex=True)

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

args = parser.parse_args()

reports_sets = dict()

# Pre-processing the files path provided regarding the comparison choice
for REPORT_PATH in args.reports:
    _, report_name = os.path.split(REPORT_PATH)

    # Skip not concerned files
    if report_name.endswith('.npz') or report_name.endswith('.npy'):
        continue

    problem_name = report_name.split('_')[0]
    sampling = report_name.split('_')[-1]

    # Aggregate the reports path according to the problem tested
    if problem_name not in reports_sets.keys():
        reports_sets[problem_name] = [REPORT_PATH]
    else:
        reports_sets[problem_name].append(REPORT_PATH)

# Go through all the reports
for operator, reports_paths in reports_sets.items():

    # Browse through the reports file names in the report set
    for j, REPORT_PATH in enumerate(reports_paths):
        pyplot.figure()
        pyplot.grid()

        metadata, deterministic_data, stochastic_data = read_report(REPORT_PATH)

        pyplot.hlines(y=metadata['reference'],
                      xmin=stochastic_data['sizes'][0] / metadata['size'],
                      xmax=stochastic_data['sizes'][-1] / metadata['size'],
                      linestyle='dashed',
                      label='CG with only first-level preconditioner.')

        pyplot.errorbar(stochastic_data['sizes'] / metadata['size'],
                        stochastic_data['means'] * metadata['reference'],
                        yerr=stochastic_data['stdevs'] * metadata['reference'],
                        fmt='o-',
                        mec='k',
                        ms=4,
                        elinewidth=0.5,
                        ecolor='k',
                        capsize=2.5,
                        label='Random sparse subspace: $\\mu \\pm \\sigma$.')

        pyplot.plot(stochastic_data['sizes'] / metadata['size'],
                    deterministic_data['0'] * metadata['reference'],
                    color=COLORS[j % len(COLORS) + 1],
                    marker='^',
                    mec='k',
                    ms=4,
                    label='Informed descent directions.')

        pyplot.plot(stochastic_data['sizes'] / metadata['size'],
                    deterministic_data['1'] * metadata['reference'],
                    color=COLORS[j % len(COLORS) + 2],
                    marker='<',
                    mec='k',
                    ms=4,
                    label='Ritz vectors associated with highest Ritz values.')

        # Set corresponding title regarding the comparison criterion
        plot_title = 'LMP results on $A=$ {}, $n=$ {}, $M=$ {}'\
            .format(operator, metadata['size'], metadata['first_lvl_preconditioner'].capitalize())

        # Add the legend, title, and axis titles
        pyplot.legend()
        pyplot.title(plot_title)
        pyplot.xlabel('Subspace size as a fraction of $n$.')
        pyplot.ylabel('Preconditioned CG number of iterations.')
        pyplot.xlim(stochastic_data['sizes'][0] / metadata['size'],
                    stochastic_data['sizes'][-1] / metadata['size'])
        pyplot.ylim(bottom=0.)

pyplot.show()
