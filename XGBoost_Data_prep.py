""" 
Data Preparation for Gradient Boosting

 XGBoost models represent all problems as a regression predictive modeling problem 
 that only takes numerical values as input. If your data is in a different form, it 
 must be prepared into the expected format.

"""
import pandas as pd
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Loading the dataset(Iris Flower)
dt = pd.read_csv("IRIS.csv")
dt.head(5)
print(dt.shape)
dtset = dt.values
print(dt.columns)
# spliting the loaded Dataset
x = dtset[:,0:4]
print(x)
y = dtset[:,4]
print(y)

# Encoing the string class into integer class
le = LabelEncoder()
le = le.fit(y)
le.y=le.transform(y)
seed = 7
test_size = 0.33
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,le.y,test_size=test_size,random_state=seed)


# Fitting Model

model = xgboost.XGBClassifier()
model.fit(x_train,y_train)
print(model)

# Making predictions
ypred = model.predict(x_test)
predictions = [round(value) for value in ypred]

# Evaluating Predictions
accuracy = accuracy_score(y_test,predictions)
print(f"Accuracy is: {accuracy*100.0:.2f}%")





