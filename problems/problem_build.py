#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 21, 2019 at 10:38.

@author: a.scotto

Description:
"""

import pickle
import scipy.io

file_name = ''

problem = dict()

obj = scipy.io.loadmat(file_name)['Problem'][0][0][1]
# for i, obj_i in enumerate(obj):
#     print('Element', i, ':', obj_i)
#     print()
problem['operator'] = obj

problem['name'] = file_name
problem['non_zeros'] = len(obj.indices)
problem['source'] = 'Structural Problem'
problem['conditioning'] = 2.272772e6
problem['symmetric'] = True
problem['def_pos'] = True

obj = scipy.io.loadmat(file_name + '_SVD')['S'][0][0][0]
# for i, obj_i in enumerate(obj):
#     print('Element', i, ':', obj_i)
#     print()
problem['singular_values'] = obj
problem['rank'] = len(obj)
problem['shape'] = (len(obj), len(obj))

print(problem)

# with open(file_name, 'wb') as file:
#
#     p = pickle.Pickler(file)
#     p.dump(problem)
