# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import Required Libraries
 Import pandas, numpy, matplotlib, and sklearn libraries for data processing, visualization, model training, and error calculation.

2.Load the Dataset
 Read the CSV file containing student study hours and scores using pandas.
 
3.Display Dataset
   View the first few rows and last few rows of the dataset to understand the data.
   Separate Independent and Dependent Variables
   Store study hours as independent variable X.
   Store student scores as dependent variable Y.

4.Split Dataset into Training and Testing Sets
 Divide the data into training set and testing set.

5.Training data → Used to train the model

6.Testing data → Used to test the model accuracy

7.Create Linear Regression Model
 Create an object for Linear Regression model.

8.Train the Model
 Train the model using training data (X_train and Y_train).

9.Predict Test Results
 Use the trained model to predict scores for testing data (X_test).

10.Visualize Training Set Result

 Plot scatter graph of training data.

 Plot regression line (best fit line).

11.Visualize Testing Set Result

 Plot scatter graph of testing data.

 Plot regression line using trained model.

12.Calculate Error Metrics
 Calculate model performance using:

 Mean Squared Error (MSE)

 Mean Absolute Error (MAE)

 Root Mean Squared Error (RMSE)

13.Display Error Values
 Print MSE, MAE, and RMSE values.

## Program :
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: PANDEESWARAN N
RegisterNumber: 212224230191
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('drive/MyDrive/Dataset-ML/student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
X,Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
Y_pred,Y_test
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Hours VS Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title('Hours VS Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("MSE = ",mse)
print('MAE = ',mae)
print('RMSE = ',rmse)

```

## Output :

<img width="366" height="234" alt="image" src="https://github.com/user-attachments/assets/4e67cbd6-8934-4393-8c0d-feff72351870" />


<img width="415" height="238" alt="image" src="https://github.com/user-attachments/assets/de0f4641-b86f-4564-bef2-629a2f0ada8c" />


<img width="1047" height="601" alt="image" src="https://github.com/user-attachments/assets/b81d5d77-e60c-44f1-ab64-9bb8a4c9e6d5" />




<img width="844" height="96" alt="image" src="https://github.com/user-attachments/assets/699fae34-7aa5-401c-bfc7-f368863eaf13" />




<img width="956" height="600" alt="image" src="https://github.com/user-attachments/assets/79ce8ef5-e191-4542-9b8c-76fb9590e961" />




<img width="912" height="593" alt="image" src="https://github.com/user-attachments/assets/434099c9-4e08-481a-8ca0-daa01e057b7c" />




<img width="337" height="88" alt="image" src="https://github.com/user-attachments/assets/c07bc482-e931-4a31-9d46-f9b9a33f219b" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
