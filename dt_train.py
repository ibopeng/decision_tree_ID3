"""
Decision tree learning algorithm, ID3

Train a decision tree using training data

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import dt_func as dt
import scipy.io.arff as af

""" Run the decision tree ID3 using terminal command line"""
# get arguments from terminal command line
# 4 arguments: (1) dt-learn.py (2) <train-set-file> (3) <test-set-file> (4) m
filename_trn, filename_test, m = dt.read_cmdln_arg()

""" 
Step 1: build decision tree
"""
# load training data
instance_data_trn, meta_data = af.loadarff('credit_train.arff')
var_ranges = [meta_data[name][1] for name in meta_data.names()]

# extract some features from meta-data
num_var = len(meta_data.types()) # number of variables of instance data, including the label
var_types = meta_data.types() # 'nominal' or 'numeric'
var_names = meta_data.names() # name of each variable
label_range = var_ranges[-1] # the range of instances labels, i.e., '+' and '-'

# build the tree
dt_root = dt.makeSubtree(instance_data_trn, label_range, num_var, var_types, var_names, var_ranges, None)
# print the tree
dt_root.print_tree()

"""
Step 2: classify the test data using trained decision tree
"""
# load test data
instance_data_test, meta_data = af.loadarff('credit_test.arff')
# extract test data true label
test_label = [ins[-1] for ins in instance_data_test]

# classify the test data
test_prediction = []
num_correct_pred = 0 # compute the number of correct predictions
num_instance_test = len(instance_data_test) # number of instances in test data

# predict each instance one by one
for i in range(num_instance_test):
    test_prediction.append(dt.dt_prediction(dt_root, instance_data_test[i], var_ranges))
    # count correct prediction
    if test_prediction[i] == test_label[i]:
        num_correct_pred += 1
    print('{0}: Actual: {1} Predicted: {2}'.format(i+1, test_label[i], test_prediction[i]) )

print('Number of correctly classified: {0} Total number of test instances: {1}'.format(num_correct_pred, num_instance_test))








