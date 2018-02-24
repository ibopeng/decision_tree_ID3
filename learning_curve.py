"""
Project: Decision tree learning algorithm, ID3

Learning curve that characterizes the predictive accuracy of your learned trees as a function of the training set size

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import dt_func as dt
import tree_node as tn
import scipy.io.arff as af
import random
import numpy as np
import matplotlib.pyplot as plt




""" 
Step 1: build decision tree
"""
# load training data
instance_data_trn, meta_data = af.loadarff('credit_train.arff')
# load test data
instance_data_test, meta_data = af.loadarff('credit_test.arff')
# extract test data true label
test_label = [ins[-1] for ins in instance_data_test]
num_instance_test = len(instance_data_test)  # number of instances in test data

# extract some features from meta-data
num_var = len(meta_data.types()) # number of variables of instance data, including the label
var_types = meta_data.types() # 'nominal' or 'numeric'
var_names = meta_data.names() # name of each variable
var_ranges = [meta_data[name][1] for name in meta_data.names()]
label_range = var_ranges[-1] # the range of instances labels, i.e., '+' and '-'

# define the training set size proportion
full_set_size = len(instance_data_trn)
train_set_percent = [5, 10, 20, 50, 100]
train_set_size = [int(full_set_size * percent / 100) for percent in train_set_percent]

num_draw = 10  # randomly draw the subset 10 times
acc_subset_size = []
for sub_size in train_set_size:
    # random sample training subset
    acc_sub_draw = []
    for i in range(num_draw):
        # get the random index
        idx = random.sample(range(full_set_size), sub_size)
        # get subset based on random index
        subset_trn = [instance_data_trn[k] for k in idx]

        # build the tree
        m = 10
        dt_root = tn.TreeNode()
        dt_root = dt.makeSubtree(subset_trn, label_range, num_var, var_types, var_names, var_ranges, None, m)
        # print the tree
        #dt_root.print_tree()

        # classify the test data
        test_prediction = [dt.dt_prediction(dt_root, instance_test, var_ranges) for instance_test in instance_data_test]
        # compute number of correct prediction
        num_correct_pred = dt.comp_num_correct_predict(test_label, test_prediction)
        accuracy = 1.0 * num_correct_pred / num_instance_test

        # record the accuracy for each random draw
        acc_sub_draw.append(accuracy)

        if sub_size == full_set_size:
            i = num_draw

    # for each size, compute the average, minimum, and maximum accuracy
    acc_sub_draw = np.array(acc_sub_draw)
    avg_acc_sub_draw = np.average(acc_sub_draw)
    min_acc_sub_draw = np.min(acc_sub_draw)
    max_acc_sub_draw = np.max(acc_sub_draw)

    # record accuracy for each size, note that for each size, there are num_draw times accuracy stored in acc_subset_rand_draw
    acc_subset_size.append([avg_acc_sub_draw, min_acc_sub_draw, max_acc_sub_draw])

# plot the learning curve
plt.plot(train_set_percent, np.array([acc_avg[0] for acc_avg in acc_subset_size]), label = 'Average')
plt.plot(train_set_percent, np.array([acc_avg[1] for acc_avg in acc_subset_size]), label = 'Min')
plt.plot(train_set_percent, np.array([acc_avg[2] for acc_avg in acc_subset_size]), label = 'Max')
plt.legend()
plt.grid()
plt.title('Learning curve for credit data')
plt.xlabel('Proportion of full training set')
plt.ylabel('Accuracy for testing set')
plt.show()
