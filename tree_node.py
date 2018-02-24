"""
Project: Decision tree learning algorithm, ID3

Class to store tree node related information

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""


class TreeNode:
    """
    tree_node class, storing information of current node
    """

    parent = None # parent of current node, only 1 parent
    children = []  # children list of current node, there may be multiple children
    major_label = None # the majority class/label of instances in current node
    node_type = None # nominal or numeric
    var_name = None # name of the node variable, A1, A2, A14...
    num_label_0 = 0  # number of instances belong to label 0, i.e., '-'
    num_label_1 = 0  # number of instances belong to label 1, i.e., '+'
    is_leaf = False # check if it is a leaf or an internal node
    threshold = 0 # if this is a numeric node, determine its threshold for best split

    # a string for printing out the tree, like '<= 2.00000 [39 175]'
    # indicating which branch, threshold if any or nominal value, number of positive and negtive instances
    branch = None
    depth = 0 # depth of current node in the decision tree
    var_idx = 0 # index of current node variable

    # print out the trained tree
    def print_tree(self):
        node_var_name = str(self.var_name).replace('A', 'a') # lower case the original variable name

        # print out the tree is actually printing out the branches associated with the children of current node
        for child in self.children:
            if not child.is_leaf:
                print(self.depth * '|\t' + node_var_name + ' ' + child.branch + ' [' + str(
                    child.num_label_0) + ' ' + str(child.num_label_1) + ']')
            # if the child of current node is a leaf, print out the major label
            else:
                print(self.depth * '|\t' + node_var_name + ' ' + child.branch + ' [' + str(
                    child.num_label_0) + ' ' + str(child.num_label_1) + ']: ' + child.major_label)
            child.print_tree()