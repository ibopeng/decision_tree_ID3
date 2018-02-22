"""
Decision tree learning algorithm, ID3

Load data file

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

def extract_attribute_name(str_attribute):
    # e.g. str_attribute = "@attribute 'A15' { b, a }"
    att_start_idx = str_attribute.find("'") + 1  # the index of the 1st '
    att_length = str_attribute[att_start_idx+1:].find("'") + 1 # the length of the attribute name
    return str_attribute[att_start_idx:att_start_idx+att_length]

def extract_attribute_value(str_attribute):
    str_attribute_array = []

    att_start_idx = str_attribute.find("{")  # the index of {
    if att_start_idx != -1:
        att_end_idx = str_attribute.find("}")  # the index of the }
        for i in range(att_start_idx+1, att_end_idx):
            char = str_attribute[i]
            if char not in [' ', ',']:
                str_attribute_array.append(char)
    else:
        str_attribute_array.append('real')
    return str_attribute_array

def load_data(filename):

    with open(filename, 'r') as fp:
        attribute = [] # store attribute lines
        instance = [] # store instances
        bData = False # indicate whether current line is data or else

        line = fp.readline()
        while line:
            str_line = line.strip()
            if not bData:
                if not str_line.startswith("@relation"):
                    att_name = extract_attribute_name(str_line)
                    att_value= extract_attribute_value(str_line)
                    attribute.append([att_name, att_value])
                if str_line == "@data":
                    bData = True
            else:
                instance.append(line.strip())
            line = fp.readline()

    # parcel the data

    return attribute, instance


attribute, instance = load_data('credit_train.arff')


print(len(attribute[3][1]))
print("Below is the data")
print(len(instance))


str = "@attribute 'A5' { g, p, gg }"
print extract_attribute_value(str)


def comp_entropy_numeric(instance_entire, label_range, var_idx):
    """
    Compute sum of entropy of each instance subset after split for numeric variable
    :param instance_data:
    :param var_idx:
    :return:
    """
    # if len(data_) == 0: raise Exception('ERROR: the input data set is empty')
    # extract the value of the ith variable, i = var_idx
    var_val = [ins[var_idx] for ins in instance_entire]
    # find median point of this numeric array for the ith variable, i = var_idx
    var_median = np.median(np.array(var_val))

    instance_split = split_data_numeric(instance_entire, var_idx, var_median)

    entropy = 0

    for instance_sub in instance_split:  # loop to compute entropy of each instance subset
        # collect all class labels
        classLabels = [ins[-1] for ins in instance_sub]
        # weight the entropy by the occurence
        p_x = 1.0 * len(instance_sub) / len(instance_entire)
        entropy += p_x * comp_entropy(classLabels, label_range)

    return entropy



def comp_entropy_nominal(instance_entire, label_range, var_idx, this_var_range):
    """
    Compute the sum of entropy of each instance subset for nominal variables
    :param instance_entire:
    :param label_range:
    :param var_idx:
    :param this_var_range:
    :return:
    """
    if len(instance_entire) == 0:
        return 0
    entropy = 0

    # split the data with the ith attribute/variable, i = var_idx
    instance_split = split_data_nominal(instance_entire, var_idx, this_var_range)

    for instance_sub in instance_split: # loop to compute entropy of each instance subset
        # collect all class labels
        classLabels = [ins[-1] for ins in instance_sub]
        # weight the entropy by the occurence
        p_x = 1.0 * len(instance_sub) / len(instance_entire)
        entropy += p_x * comp_entropy(classLabels, label_range)

    return entropy


# extract the value of the ith variable, i = var_idx
var_val = [ins[var_idx] for ins in instance_data]
# find median point of this numeric array for the ith variable, i = var_idx
var_median = np.median(np.array(var_val))
# split data, median point is set to be the threshold
instance_split = split_data_numeric(instance_data, var_idx, var_median)
# compute the sum of entropy of each instance subset after split
entropy = comp_entropy_split_post(instance_split, instance_data, label_range)