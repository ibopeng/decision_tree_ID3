"""
Decision tree learning algorithm, ID3

Train a decision tree using training data

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import load_data as ld
import decision_tree as dt
import numpy as np

""" Load training data, and corresponding parameters"""
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

instance_data_test, meta_data, var_ranges, var_unique_val = ld.load_data('credit_test.arff')

prediction = []
for instance in instance_data_test:
    prediction.append(dt.dt_prediction(dt_root, instance, var_ranges))

print(prediction)

test_label = [ins[-1] for ins in instance_data_test]

accuracy = dt.comp_accuracy_test(test_label, prediction)

print accuracy








