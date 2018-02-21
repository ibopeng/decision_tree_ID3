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
