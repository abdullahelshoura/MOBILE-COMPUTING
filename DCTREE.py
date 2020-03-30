
#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#2 Importing the dataset
path = "diabetes.csv"
dataset = pd.read_csv(path)
X = dataset.iloc[:, :8].values
y = dataset.iloc[:, 8].values
print(dataset)

#3 Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
#4 Create Decision Tree classifer object
classifier = DecisionTreeClassifier()

#5 Train Decision Tree Classifer
classifier.fit(X_train, y_train)

#6 Predict the response for test dataset
y_pred = classifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
