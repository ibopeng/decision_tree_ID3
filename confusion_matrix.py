import numpy as np
from sklearn.metrics import confusion_matrix

# load data
data = np.loadtxt('predictions.txt', dtype=np.int32, delimiter=',')

# extract predicted class and true class for each instance
y_pred = np.array([data[i][0] for i in range(len(data))])
y_true = np.array([data[i][1] for i in range(len(data))])

# compute confusion matrix
conf_mat = confusion_matrix(y_pred, y_true)

# print out the result
print("Every row indicates true class whereas column indicates predicted class")
print(conf_mat)