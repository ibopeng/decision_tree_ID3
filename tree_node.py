"""
Project: Decision tree learning algorithm, ID3

Class to store tree node related information

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

class tree_node:
    # define the parent and children of this node
    parent = None
    children = []
    major_label = None
    node_type = None
    var_name = None
    num_label_0 = 0  # number of instances belong to label 0, i.e., '-'
    num_label_1 = 0  # number of instances belong to label 1, i.e., '+'
    is_leaf = False
    threshold = 0
    branch = None
