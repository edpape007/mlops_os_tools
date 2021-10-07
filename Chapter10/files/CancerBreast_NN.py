import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow
from keras.models import Sequential
from keras.layers import Dense



colnames = ['ID', 'Diag', 'radius', 'texture','perimeter', 'area', 'smoothness', 'compactness', 'concavity', 
                'concave_points', 'symmetry', 'fractal_dim', 'se_radius', 'se_texture','se_perim', 'se_area', 'se_smooth', 
                'se_compact', 'se_concavity', 'se_concpts', 'se_symm', 'se_fractdim', 'max_rad', 'max_text', 'max_perim', 
                'max_area', 'max_smooth', 'max_compact', 'max_concavity', 'max_concpts', 'max_symm', 'max_fracdim']

data = pd.read_csv('/Users/edpape/miniforge3/envs/dags/files/wdbc.data', header = None, names = colnames)
data = data.iloc[:, 1:]
data = pd.get_dummies(data, columns=['Diag'])

X = data.iloc[:, :30]
y = data.iloc[:, 30:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train = np.asmatrix(X_train)
X_train = preprocessing.normalize(X_train)

X_test = np.asmatrix(X_test)
X_test = preprocessing.normalize(X_test)

y_train = np.asmatrix(y_train)
y_test = np.asmatrix(y_test)

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

print("Dimension Y Train")
print(y_train.shape)
print("Dimension Y Test")
print(y_test.shape)

# define the keras model
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# # compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # fit the keras model on the dataset
model.fit(X_train, y_train, epochs=500, batch_size=10, verbose=0)
# # evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))