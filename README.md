# multi_linear_re_50_startup02
his project implements multi-linear regression analysis on a dataset comprising financial information of 50 startups. The goal is to predict the profit based on various independent variables such as R&amp;D spend, administration spend, and marketing spend.


import pandas as pd  

dataset = pd.read_csv("50_Startups.csv")
dataset.info()
dataset.head()
dataset.shape
dataset.isnull().sum()
dataset.columns
y = dataset['Profit']
x = dataset[['R&D Spend','Administration','Marketing Spend','State']]
y 
state = dataset["State"]
pd.get_dummies(state) 
state_dummy = pd.get_dummies(state)
type(state_dummy)
final_dummy_variable = state_dummy.iloc[ : , 0:2] 
y = dataset['Profit']
x = dataset[['R&D Spend','Administration','Marketing Spend']]
X = pd.concat([x , final_dummy_variable],axis=1)
X
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test= train_test_split(X,y ,test_size=0.20 , random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_preadict = model.predict(X_test)
y_preadict
X_test
y_test
model.fit(X,y)
from sklearn import metrics
metrics.mean_absolute_error(y_test,y_preadict)
model.coef_
model.intercept_
