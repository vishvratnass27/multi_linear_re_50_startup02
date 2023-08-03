# multi_linear_re_50_startup02
his project implements multi-linear regression analysis on a dataset comprising financial information of 50 startups. The goal is to predict the profit based on various independent variables such as R&amp;D spend, administration spend, and marketing spend.


**Step 1: Importing Libraries**
```python
import pandas as pd
```
This line imports the pandas library and gives it the alias "pd" to use later in the code. Pandas is used for data manipulation and analysis.

**Step 2: Loading the Dataset**
```python
dataset = pd.read_csv("50_Startups.csv")
```
This line reads the CSV file named "50_Startups.csv" and stores it in the variable `dataset`. The CSV file contains data about startups, including their R&D Spend, Administration, Marketing Spend, State, and Profit.

**Step 3: Exploring the Dataset**
```python
dataset.info()
dataset
dataset.columns
```
These lines show information about the dataset, such as the data types of columns and the number of non-null values. It also displays the entire dataset and the names of its columns.

**Step 4: Preparing the Data**
```python
y = dataset['Profit']
x = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
```
Here, we separate the dependent variable `y`, which is the Profit column, and the independent variables `x`, which include R&D Spend, Administration, Marketing Spend, and State.

**Step 5: Handling Categorical Data**
```python
state = dataset["State"]
pd.get_dummies(state)
state_dummy = pd.get_dummies(state)
final_dummy_variable = state_dummy.iloc[:, 0:2]
X = pd.concat([x, final_dummy_variable], axis=1)
```
The State column is a categorical variable, and we need to convert it into numerical values. We use the one-hot encoding method with `pd.get_dummies()` to create dummy variables for each category in the State column. Then, we concatenate these dummy variables with the independent variables `x` to form the final input data `X`.

**Step 6: Splitting the Data**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
```
We split the data into training and testing sets using the `train_test_split()` function from scikit-learn. The training set (X_train and y_train) will be used to train the linear regression model, and the testing set (X_test and y_test) will be used to evaluate the model's performance.

**Step 7: Creating and Training the Linear Regression Model**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
We import the LinearRegression class from scikit-learn and create a linear regression model named `model`. Then, we train the model using the training data (X_train and y_train) with the `fit()` method.

**Step 8: Making Predictions**
```python
y_predict = model.predict(X_test)
```
We use the trained model to make predictions on the testing data, and the predictions are stored in `y_predict`.

**Step 9: Evaluating the Model**
```python
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_predict)
```
Finally, we import the `metrics` module from scikit-learn and calculate the mean absolute error between the actual profit values (`y_test`) and the predicted profit values (`y_predict`). This metric helps us evaluate how well the model performs.

That's the basic explanation of the code! Linear regression is a simple but powerful algorithm used for predicting numerical values. I hope this helps you understand the code better! If you have any more questions, feel free to ask. Happy learning!
