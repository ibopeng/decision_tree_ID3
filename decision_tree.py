"""
Project: Decision tree learning algorithm, ID3

Build a ID3 like decision tree

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""
import numpy as np
import scipy as sp


def comp_entropy(instance_label, label_range):
    """
    Compute the entropy for current state
    :param instance_label: label of all instances
    :param label_range: range of the labels
    :return: entropy
    """

    # number of instances having one same class
    # select one class to count the total number, e.g., label_range[0]
    num_instance = len(instance_label)
    num_one_class = sum(np.array([label_range[0] == instance_label[i] for i in range(num_instance)]))

    # compute the entropy for current state
    if num_one_class == num_instance or num_one_class == 0:
        # if all instances belong to one class, either class 0 (label_range[0]) or class 1 (label_range[1])
        # it indicates entropy for current state is 0
        return 0
    else:
        # if there exists multiple class of instances in current state
        # the probability of one class
        p = float(num_one_class) / num_instance
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return entropy


def split_data_nominal(instance_data, var_idx, this_var_range):
    """
    Split data based on the ith variable, where i = var_idx
    :param instance_data:
    :param var_idx:
    :param this_var_range:
    :return:
    """

    instance_split = []

    for var_val in this_var_range: # loop in the var_range to find subset of instance belong to the ith variable (i = var_idx)
        instance_sub = []
        for ins in instance_data: # loop in instance_data to assign each instance to corresponding var_val
            if ins[var_idx] == var_val: # check if this instance corresponds to this variable value
                instance_sub.append(ins)
        instance_split.append(instance_sub) # save the instance subset separately
    return instance_split


def split_data_numeric(instance_data, var_idx, threshold):
    instance_sub_lessequal = []
    instance_sub_greater = []

    for ins in instance_data: # assign each instance to one of the subset, either less equal or greater than threshold
        if ins[var_idx] <= threshold:
            instance_sub_lessequal.append(ins)
        else:
            instance_sub_greater.append(ins)
    return [instance_sub_lessequal, instance_sub_greater]


def comp_entropy_split_post(instance_split, instance_entire, label_range):
    entropy = 0
    for instance_sub in instance_split: # loop to compute entropy of each instance subset
        # extract labels of each instance in the subset
        instance_sub_label = [ins[-1] for ins in instance_sub]
        # compute the weight of each instance subset after split
        w = 1.0 * len(instance_sub) / len(instance_entire)
        entropy += w * comp_entropy(instance_sub_label, label_range)

    return entropy

def numeric_split_min_entropy_threshold(instance_data, var_idx, label_range):
    # extract the value of the ith variable, i = var_idx
    var_val = [ins[var_idx] for ins in instance_data]

    # compute midpoints of adjacent sorted variable values
    var_val_sort = np.sort(np.array(var_val)) # sort numeric variable values
    midpoint_list = np.divide(np.add(var_val_sort[0:-1], var_val_sort[1:]), 2.0) # compute midpoints of adjacent values
    threshold_list = np.unique(midpoint_list) # extract unique midpoints as thresholds

    # loop over the threshold list to find the threshold which produces minimum entropy
    entropy_list = []
    for t in range(len(threshold_list)):
        # conditional entropy of the instance split corresponding to the t-th threshold
        instance_split = split_data_numeric(instance_data, var_idx, threshold_list[t])
        entropy_list.append(comp_entropy_split_post(instance_split, instance_data, label_range))

    # find the minimum entropy and its corresponding threshold
    idx_min = np.argmin(np.array(entropy_list))
    entropy_min = entropy_list[idx_min]
    threshold = threshold_list[idx_min]

    return entropy_min, threshold



def comp_conditional_entropy(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_range):
    """
    Compute conditional entropy for each variable not in the tree
    :param instance_data:
    :param label_range:
    :param var_in_tree:
    :param num_var:
    :param var_types:
    :param var_names:
    :param var_range:
    :return:
    """
    # number of variables as input for binary classification, the last variable is output "class", should be excluded
    num_var_input = num_var - 1
    # generate an array to store the entropy conditioned on each attribute after split
    entropy_split_post = np.zeros(num_var_input)
    entropy_split_post.fill(np.NaN)

    # loop to compute the entropy conditioned on each variable not in the tree after split
    for var_idx in range(num_var_input):
        if var_in_tree[var_idx] == True:# if this variable/attribute is in the tree
            continue
        else:# if this variable/attribute is not in the tree
            if var_types[var_idx] == 'nominal':# if this is a nominal attribute/variable
                # split data
                instance_split = split_data_nominal(instance_data, var_idx, var_range[var_idx])
                # compute the sum of entropy of each instance subset after split
                entropy = comp_entropy_split_post(instance_split, instance_data, label_range)
#                entropy = comp_entropy_nominal(instance_data, label_range, var_idx, var_range[var_idx])
            else:# if this is a numeric attribute/variable
                entropy, threshold = numeric_split_min_entropy_threshold(instance_data, var_idx, label_range)

        # save the entropy
        entropy_split_post[var_idx] = entropy
    return entropy_split_post


def comp_info_gain(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_range):
    # compute information gain for all features
    # extract labels for each instance
    instance_label = [instance_data[i][-1] for i in range(len(instance_data))]
    # compute entropy before split
    entropy_split_pre = comp_entropy(instance_label, label_range)
    # compute conditional entropy after split
    entropy_split_post = comp_conditional_entropy(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_range)
    info_gain = np.subtract(entropy_split_pre, entropy_split_post)

    return info_gain

def is_node_same_class(instance_data):
    """
    Check if all of the training instances reaching the node belong to the same class
    :param instance_data:
    :return: True if all same class, otherwise False
    """
    # extract labels of the instance data
    instance_label = [ins[-1] for ins in instance_data]

    # if all instance of this node belong to the same class, the number of unique values of labels should be 1
    if len(set(instance_label)) == 1: # "set" is a function for Unordered collections of unique elements
        return True

    return False

def stop_grow_tree(instance_data, info_gain, var_in_tree, m):
    """
    Check if to stop growing the subtree
    :param instance_data:
    :param info_gain:
    :param var_in_tree:
    :param m:
    :return:
    """
    # note that the array [info_gain] contains np.nan value
    # extract non-nan value in "info_gain" array
    info_gain = np.array(info_gain)
    info_gain = info_gain[~np.isnan(info_gain)]

    # One of the following criteria should be met to stop growing the tree
    # (1) all of the training instances reaching the node belong to the same class, or
    # (2) there are fewer than m training instances reaching the node, where m is provided as input to the program, or
    # (3) no feature has positive information gain, or
    # (4) there are no more remaining candidate splits at the node.
    if is_node_same_class(instance_data) or len(instance_data) < m or all(sp.less_equal(info_gain, 0)) or all(var_in_tree):
        return True

    return False


def makeSubtree(instance_data, label_range, var_name_tree, var_in_tree, var_val_cur, num_var, var_types, var_names, var_range):
    """
    Build the decision tree
    :param instance_data_trn: training data, including multiple instances
    :param var_name_tree:
    :param var_in_tree: variable already put as a node in the tree
    :param var_val_cur:
    :return: decision_tree_ID3: a sub tree
    """

    decision_tree_ID3 = 47 # delete this line when completing the program

    # compute the current information gain to check if stop or not
    info_gain = comp_info_gain(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_range)

    if stop_grow_tree(instance_data, info_gain, var_in_tree, 5):
        is_leaf = True
        return is_leaf


    return decision_tree_ID3

