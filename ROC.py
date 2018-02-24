"""
Project: Decision tree learning algorithm, ID3

ROC for decision tree

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import scipy.io.arff as af
import dt_func as dt
import numpy as np
import matplotlib.pyplot as plt


def dt_confidence(node, instance, var_ranges):
    """
    # compute the confidence of this instance
    :param node: root node of the trained decision tree
    :param instance: a single instance
    :param var_ranges: [list] possible values of each variable
    :return: confidence of this instance
    """

    # the major label of the leaf node is returned as the predicted label
    if node.is_leaf:
        num_pos = node.num_label_0
        num_neg = node.num_label_1
        confidence = 1.0 * (num_pos + 1) / (num_pos + num_neg + 1 + 1)
        return confidence
    else:
        # search along the decision tree
        var_idx = node.var_idx  # get the index of the variable in instance
        if node.node_type == 'numeric':
            if instance[var_idx] <= node.threshold:
                node_next = node.children[0]
            else:
                node_next = node.children[1]
        else:
            this_var_range = var_ranges[var_idx]
            for i in range(len(this_var_range)):
                if instance[var_idx] == this_var_range[i]:
                    node_next = node.children[i]
        return dt_confidence(node_next, instance, var_ranges)


def ROC(test_confidence, test_label):
    """
    Compute the ROC array
    :param test_confidence: confidence of test instances
    :param test_label: true label of test instances
    :return: ROC array
    """

    test_confidence = np.array(test_confidence)

    # compute actual number of positive and negative instances
    num_instance = len(test_confidence)
    num_true_pos = sum(np.array(['+' == test_label[i] for i in range(num_instance)]))
    num_true_neg = num_instance - num_true_pos

    # for each threshold, compute the TP and FP
    ROC_array = []

    zipped = zip(test_confidence, test_label)
    zipped.sort(key = lambda t: t[0]) # sort confidence and label based on confidence, ascending order
    zipped.reverse() # sort the confidence from high to low, descending order
    [test_confidence, test_label] = zip(*zipped)
#    test_confidence = [zip_tuple[0] for zip_tuple in zipped]
#    test_label = [zip_tuple[1] for zip_tuple in zipped]

    cutoff = []
    cutoff.append(1)
    for i in range(num_instance):
        if i == 0:
            cutoff.append(test_confidence[0])
            current_state = test_label[0]
        else:
            if current_state == test_label[i]:
                continue
            else:
                current_state = test_label[i]
                cutoff.append(test_confidence[i-1])
                cutoff.append(test_confidence[i])
    cutoff.append(0)

    for cf in cutoff:
        # compute true positive and false positive
        TP = 0
        FP = 0
        for i in range(num_instance):
            if test_confidence[i] < cf:
                break
            else:
                if test_label[i] == '+':
                    TP += 1
                elif test_label[i] == '-':
                    FP += 1
        TP_rate = 1.0 * TP / num_true_pos
        FP_rate = 1.0 * FP / num_true_neg
        ROC_array.append([FP_rate, TP_rate])

    return ROC_array


# load training data
instance_data_trn, meta_data = af.loadarff('credit_train.arff')
var_ranges = [meta_data[name][1] for name in meta_data.names()]

# extract some features from meta-data
num_var = len(meta_data.types()) # number of variables of instance data, including the label
var_types = meta_data.types() # 'nominal' or 'numeric'
var_names = meta_data.names() # name of each variable
label_range = var_ranges[-1] # the range of instances labels, i.e., '+' and '-'

# build the tree
dt_root = dt.makeSubtree(instance_data_trn, label_range, num_var, var_types, var_names, var_ranges, None, 10)
# print the tree
dt_root.print_tree()

# load test data
instance_data_test, meta_data = af.loadarff('credit_test.arff')
# extract test data true label
test_label = [ins[-1] for ins in instance_data_test]
num_instance_test = len(instance_data_test)  # number of instances in test data

# compute the confidence for test instance data
test_confidence = [dt_confidence(dt_root, instance_test, var_ranges) for instance_test in instance_data_test]
threshold_list = np.sort(np.array(test_confidence))
np.savetxt('confidence.txt', threshold_list)

ROC_array = ROC(test_confidence, test_label)

[x_FP, y_TP] = zip(*ROC_array)
plt.plot(x_FP, y_TP)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC')
plt.grid()
plt.show()