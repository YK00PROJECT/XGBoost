"""
XGBoost is an implementation of gradient boosted decision trees designed for speed and performance
Using XGBoost to predict Onset of Diabetes.
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, 
based on certain diagnostic measurements included in the dataset.

"""

# The loadtxt function specifically is used to load data from a text file
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the dataset (Pima Indians Diabetes Database)
dt =loadtxt('diabetes.csv', delimiter=',',skiprows=1)

# Spliting the dataset(dt) into x and y
x =dt[:,0:8]
y = dt[:,8]

# Spliting Data into train and test sets
seed = 7
test_size = 0.33
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=seed)

"""
XGBoost includes a wrapper class that enables its models to function as either classifiers 
or regressors within the scikit-learn ecosystem.
This allows for the integration of XGBoost models with the comprehensive scikit-learn library.
"""

# Training The XGBoost Model on the training dataset

model = XGBClassifier()
model.fit(x_train,y_train)

print(model)

# Making prediction for test data
ypred = model.predict(x_test)
predictions = [round(value) for value in ypred]

# Evaluating Predictions

accuracy = accuracy_score(y_test,predictions)
print('Accuracy is : %.2f%%' % (accuracy * 100.0))



