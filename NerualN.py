#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

#preparing ( splitting .... ) training,testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#training the model
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4), random_state=1)#one hidden layer with 4 neurons
classifier.fit(X_train, y_train)

#mpredictions to test the model
y_pred = classifier.predict(X_test)

#printing the weights
print("the coefs are : ",classifier.coefs_)

#printing the bias
print("\n the biases are : ",classifier.intercepts_)
   
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))