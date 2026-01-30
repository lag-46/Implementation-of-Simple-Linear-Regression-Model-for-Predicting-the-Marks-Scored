# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

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
