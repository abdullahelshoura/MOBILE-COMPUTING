#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

path = "diabetes.csv"
dataset = pd.read_csv(path)
X = dataset.iloc[:, :8].values
y = dataset.iloc[:, 8].values
print(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#Create a Gaussian Classifier
classifier = RandomForestClassifier(n_estimators = 50) #n_estimator represent number of decision tree

#Train the model using the training sets y_pred=clf.predict(X_test)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


#showing in figure if the predicted value is exist or not ..if it is it gives a pulse (1) if not it is nothing (0)
size=y_pred.size
fir=range(0,size)
sec=[]
for i in range(size):
    if y_test[i]==y_pred[i]:
        sec.append(1)
    else:
        sec.append(0)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
plt.plot(fir,sec)
plt.show()
