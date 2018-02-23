"""
Project: Decision tree learning algorithm, ID3

Main script for running ID3

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

# import packages
import load_data as ld
import decision_tree as dt
import numpy as np



""" Run the decision tree ID3 using terminal command line"""
# get arguments from terminal command line
# 4 arguments: (1) dt-learn.py (2) <train-set-file> (3) <test-set-file> (4) m
#filename_trn, filename_test, m = ld.read_cmdln_arg()

""" Load training data and testing data, and corresponding parameters"""
instance_data_trn, meta_data, var_ranges, var_unique_val = ld.load_data('credit_train.arff')


# variables already in tree or not, bool type, True and False
num_var = len(meta_data.types())
var_types = meta_data.types()
var_names = meta_data.names()

var_in_tree = np.zeros(num_var - 1, dtype=bool)
# the range of instances labels, i.e., '+' and '-'
label_range = var_ranges[-1]
# name of each variable
var_names = meta_data.names()

# build the tree
dt_root = dt.makeSubtree(instance_data_trn, label_range, var_in_tree, num_var, var_types, var_names, var_ranges, None)

dt_root.print_tree()

print("Well done...")

