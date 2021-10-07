import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

colnames = ['ID', 'Diag', 'radius', 'texture','perimeter', 'area', 'smoothness', 'compactness', 'concavity', 
                   'concave_points', 'symmetry', 'fractal_dim', 'se_radius', 'se_texture','se_perim', 'se_area', 'se_smooth', 
                   'se_compact', 'se_concavity', 'se_concpts', 'se_symm', 'se_fractdim', 'max_rad', 'max_text', 'max_perim', 
                   'max_area', 'max_smooth', 'max_compact', 'max_concavity', 'max_concpts', 'max_symm', 'max_fracdim']

data = pd.read_csv('/Users/edpape/miniforge3/envs/dags/files/wdbc.data', header = None, names = colnames)
data = data.iloc[:, 1:]

data["Diag"].value_counts()
cleanup_nums = {"Diag":     {"B": 1, "M": 0}}
data = data.replace(cleanup_nums)

X = data.iloc[:, 1:30]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)