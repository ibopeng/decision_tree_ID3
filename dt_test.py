"""
Project: Decision tree learning algorithm, ID3

Using trained decision tree to classify the test data

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


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
