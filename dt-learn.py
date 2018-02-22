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

#############
tt = [0.8, 0.3, np.nan, 0.4, np.nan, 8.3]
tt = np.array(tt)
tn = ~np.isnan(tt)
###############

""" Run the decision tree ID3 using terminal command line"""
# get arguments from terminal command line
# 4 arguments: (1) dt-learn.py (2) <train-set-file> (3) <test-set-file> (4) m
#filename_trn, filename_test, m = ld.read_cmdln_arg()

""" Load training data and testing data, and corresponding parameters"""
instance_data_trn, meta_data, var_range, var_unique_val = ld.load_data('credit_train.arff')


# variables already in tree or not, bool type, True and False
num_var = len(meta_data.types())
var_types = meta_data.types()
var_names = meta_data.names()

var_in_tree = np.zeros(num_var - 1, dtype=bool)
# the range of instances labels, i.e., '+' and '-'
label_range = var_range[-1]
# name of each variable
var_names = meta_data.names()

# build the tree
var_val_cur = None
var_name_tree = None # this para should be taken good care of
decision_tree_ID3 = dt.makeSubtree(instance_data_trn, label_range, var_name_tree, var_in_tree, var_val_cur, num_var, var_types, var_names, var_range)



print("Debug...")

