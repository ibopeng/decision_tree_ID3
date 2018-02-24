"""
Project: Decision tree learning algorithm, ID3

Load data file

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import sys
import scipy.io.arff as af


def read_cmdln_arg():
    """command line operation"""
    if len(sys.argv) != 4:
        sys.exit("Incorrect arguments...")
    else:
        filename_trn = str(sys.argv[1])
        filename_test = str(sys.argv[2])
        m = int(str(sys.argv[3]))
    return filename_trn, filename_test, m


def load_data(filename):
    """
    read data in standard arff format
    Input:
        filename: the name of the data file with the format of .arff
    Output:
        instance_data: array[m,n], m = number of instances, n = number of variables
        meta_data: class object, including info about data attribute and class range
        var_range: possible value defined for each variable
        var_uniqeu_val: possible value extracted from instance_data for each variable
    """
    instance_data, meta_data = af.loadarff(filename)

    # extract range of each variable from meta_data
    var_range = []
    for name in meta_data.names():
        var_range.append(meta_data[name][1])

    return instance_data, meta_data, var_range