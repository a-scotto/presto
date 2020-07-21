#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 20, 2019 at 09:09.

@author: a.scotto

Description:
"""

import json
import numpy
import argparse

from matplotlib import pyplot
from utils.utils import merge_reports

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

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('report',
                    nargs='*',
                    help='Paths of reports text files to post-process.')
parser.add_argument('-t'
                    '--time',
                    dest='time',
                    default=True,
                    action='store_true',
                    help='Whether to plot the approximate spectrum histogram.')

args = parser.parse_args()

reps = list()

import os

for rep in os.listdir('reports/'):
    if "oilpan" in rep:
        reps.append('reports/' + rep)

args.report = reps

# Sort the reports by operator and preconditioner tested
merged_reports = merge_reports(args.report)

# Go through all the reports
for operator in merged_reports.keys():
    for preconditioner in merged_reports[operator]:
        if args.time:
            figure, (axes1, axes2) = pyplot.subplots(1, 2, figsize=(10, 5))
        else:
            figure, axes1 = pyplot.subplots(figsize=(5, 5))
            axes2 = None

        for i, REPORT_FILE in enumerate(merged_reports[operator][preconditioner]):

            with open(REPORT_FILE, 'r') as file:
                report = json.load(file)

            figure.suptitle('LMP performances vs dimension of subspace, $M=${}'
                            .format(report['preconditioner']['name'].capitalize()),
                            fontsize=14)

            x_axis = list()
            error_bar = list() if report['subspace']['n_tests'] != 1 else None
            for sub_dim, data in report['data'].items():
                x_axis.append(numpy.mean(data['iterations']))
                if error_bar is not None:
                    error_bar.append(numpy.std(data['iterations']))

            y_axis = numpy.asarray(report['computational_cost']) / report['MAX_BUDGET']

            label = report['subspace']['name'].replace('_', ' ').capitalize()
            try:
                label += ' (' + str(report['subspace']['args'][0]) + ')'
            except IndexError:
                pass

            axes1.errorbar(y_axis[:len(x_axis)], x_axis,
                           c=COLORS[i % len(COLORS)], fmt='s-', mec='k', ms=3,
                           yerr=error_bar, elinewidth=1, ecolor='k', capsize=3,
                           label=label)

            if axes2 is not None:
                x_axis = list()
                error_bar = list() if report['subspace']['n_tests'] != 1 else None
                for sub_dim, data in report['data'].items():
                    rate = numpy.asarray(data['time']) / numpy.asarray(data['iterations'])
                    x_axis.append(numpy.mean(rate))
                    if error_bar is not None:
                        error_bar.append(numpy.std(rate))

            axes2.errorbar(y_axis[:len(x_axis)], x_axis,
                           c=COLORS[i % len(COLORS)], fmt='s-', mec='k', ms=3,
                           yerr=error_bar, elinewidth=1, ecolor='k', capsize=3,
                           label=label)

        # Plot reference line
        axes1.hlines(y=report['reference'],
                     xmin=0.,
                     xmax=1.,
                     linestyle='dashed',
                     label='PCG with $M$ alone')

        # Set corresponding title regarding the comparison criterion
        plot_title = 'LMP on {} ($n=$ {})'.format(operator, report['n'])

        # Add the legend, title, and axis titles
        axes1.legend(loc=3)
        axes1.title.set_text(plot_title)
        axes1.set_xlabel('Fraction of maximum budget used.')
        axes1.set_ylabel('PCG number of iterations.')
        axes1.set_xlim(0., 1.)
        axes1.set_ylim(bottom=0.)
        axes1.grid()

pyplot.show()
