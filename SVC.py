#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

path = "diabetes.csv"
dataset = pd.read_csv(path)
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, 8].values
print(dataset)

#Define (splitting dataset into 2 parts of training set) train and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
C = 1.0

#Applying SVM on dataset
Svc_classifier = svm.SVC(kernel='linear', C=C).fit(X, y)

#prediction on test data by the model
y_pred = Svc_classifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

