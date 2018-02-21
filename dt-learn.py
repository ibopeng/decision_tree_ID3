"""
Decision tree learning algorithm, ID3

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
instance_data_trn, meta_data, var_range, var_unique_val= ld.load_data('credit_train.arff')

var_in_tree = np.zeros((len(meta_data.types()) - 1,), dtype=bool)
target_label = var_range[-1]
var_names = meta_data.names()

# build the tree
var_val_cur = None
var_name_tree = None # this para should be taken good care of
decision_tree_ID3 = dt.makeSubtree(instance_data_trn, var_name_tree, var_in_tree, var_val_cur)

print("Debug...")

