"""
Project: Decision tree learning algorithm, ID3

Build a ID3 like decision tree

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""
import numpy as np
import scipy as sp
import tree_node as tn


def comp_entropy(instance_label, label_range):
    """
    Compute the entropy for current state
    :param instance_label: label of all instances
    :param label_range: range of the labels
    :return: entropy
    """
    if len(instance_label) == 0:
        return 0
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
    if len(instance_data) == 0:
        return 0
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
    if len(instance_data) == 0:
        return 0

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


def stop_grow_tree(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_ranges, m):
    """
    Check if to stop growing the subtree
    :param instance_data:
    :param info_gain:
    :param var_in_tree:
    :param m:
    :return:
    """


    # One of the following criteria should be met to stop growing the tree
    # (1) all of the training instances reaching the node belong to the same class, or
    # (2) there are fewer than m training instances reaching the node, where m is provided as input to the program, or
    # (3) no feature has positive information gain, or
    # (4) there are no more remaining candidate splits at the node.

    if len(instance_data) < m:
        return True

    if is_node_same_class(instance_data):
        return True

    # compute information gain
    # note that the array [info_gain] contains np.nan value
    # extract non-nan value in "info_gain" array
    # compute the current information gain to check if stop or not
    info_gain = comp_info_gain(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_ranges)
    info_gain = np.array(info_gain)
    info_gain = info_gain[~np.isnan(info_gain)]
    if all(sp.less_equal(info_gain, 0)):
        return True

    return False

def num_node_instance_pos_neg(instance_data, label_range, node_parent):
    """
    Count the number of positive and negative instances in current node
    :param instance_data:
    :return:
    """
    # get the lable of instance data
    instance_label = [ins[-1] for ins in instance_data]

    # count the number of positive and negative instances in current node
    num_label_0 = sum(np.array([label_range[0] == lb for lb in instance_label]))
    num_label_1 = len(instance_label) - num_label_0

    # determine the label for this node
    if node_parent != None:
        if num_label_0 > num_label_1:
            major_label = label_range[0]
        elif num_label_0 < num_label_1:
            major_label = label_range[1]
        else:
            major_label = node_parent.major_label
    else:
        if num_label_0 >= num_label_1:
            major_label = label_range[0]
        else:
            major_label = label_range[1]

    return major_label, num_label_0, num_label_1


def makeSubtree(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_ranges, node_parent):
    """
    Build the decision tree
    :param instance_data_trn: training data, including multiple instances
    :param var_name_tree:
    :param var_in_tree: variable already put as a node in the tree
    :param var_val_cur:
    :return: decision_tree_ID3: a sub tree
    """

    # compute the current information gain to check if stop or not
#    info_gain = comp_info_gain(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_ranges)

    # create a node
    node = tn.tree_node()
    node.parent = node_parent
    node.children = []
    if node_parent is not None:
        node.depth = node_parent.depth + 1
    else:
        node.depth = 0

    # if this is a leaf node
#    if stop_grow_tree(instance_data, info_gain, var_in_tree, 5):
#        node.is_leaf = True
#        node.major_label, node.num_label_0, node.num_label_1 = num_node_instance_pos_neg(instance_data, label_range, node_parent)
#        return node

    if stop_grow_tree(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_ranges, 5):
        node.is_leaf = True
        node.major_label, node.num_label_0, node.num_label_1 = num_node_instance_pos_neg(instance_data, label_range, node_parent)
        node.var_name = 'leaf'
        return node

    # if this not a leaf node, start split
    else:
        info_gain = comp_info_gain(instance_data, label_range, var_in_tree, num_var, var_types, var_names, var_ranges)
        var_split_idx = np.argmax(info_gain) # find the index of the variable for best split

        # start creating the tree

        node.is_leaf = False
        node.var_name = var_names[var_split_idx]
        node.node_type = var_types[var_split_idx]
        node.var_idx = var_split_idx
        node.major_label, node.num_label_0, node.num_label_1 = num_node_instance_pos_neg(instance_data, label_range,
                                                                                       node_parent)
        # split the data using the best variable to get best split
        # if this is a numeric variable
        if node.node_type == "numeric":
            # determine the threshold for numeric variable
            _, node.threshold = numeric_split_min_entropy_threshold(instance_data, var_split_idx, label_range)

            # split the current instance data
            [instance_sub_lessequal, instance_sub_greater] = split_data_numeric(instance_data, var_split_idx, node.threshold)

            # add child(less equal) to the parent node children list
            child_lessequal = makeSubtree(instance_sub_lessequal, label_range,  var_in_tree, num_var, var_types, var_names, var_ranges, node)
            child_lessequal.branch = "<= " + ("{0:.6f}".format(node.threshold))
            node.children.append(child_lessequal)

            # add child(greater) to the parent node children list
            child_greater = makeSubtree(instance_sub_greater, label_range, var_in_tree, num_var, var_types, var_names, var_ranges, node)
            child_greater.branch = "> " + ("{0:.6f}".format(node.threshold))
            node.children.append(child_greater)

        # if this is a nominal variable
        else:
            # get the range of this variable
            this_var_range = var_ranges[var_split_idx]
            # split the current instance data
            instance_split = split_data_nominal(instance_data, var_split_idx, this_var_range)

            # for each subset after split
            for i in range(len(instance_split)):
                child_sub = makeSubtree(instance_split[i], label_range, var_in_tree, num_var, var_types, var_names, var_ranges, node)
                child_sub.branch = "= " + str(this_var_range[i])
                node.children.append(child_sub)
#    print("Node name", node.var_name)
    return node


def dt_prediction(node, instance, var_ranges):
    if node.is_leaf:
        return node.major_label
    else:
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
        return dt_prediction(node_next, instance, var_ranges)


def comp_accuracy_test(instance_data_label, instance_data_prediction):

    if len(instance_data_label) == len(instance_data_prediction):
        num_instance = len(instance_data_label)
        num_correct_pred = sum(np.array([instance_data_label[i] == instance_data_prediction[i] for i in range(num_instance)]))
        return num_correct_pred
    else:
        return 0