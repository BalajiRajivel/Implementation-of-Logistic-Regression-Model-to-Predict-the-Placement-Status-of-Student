# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
## Developed by: BALAJI R
## RegisterNumber: 212222040023

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```


## Output:
## Placement Data:
![Screenshot 2024-03-12 094055](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/73d53fd7-16f4-4cd7-96a3-22cb95f2e2eb)

## Salary Data:
![Screenshot 2024-03-12 094106](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/ec5697be-43ea-41d2-8239-4ddf15efb1b3)

## Checking the null() function: 
![Screenshot 2024-03-12 094146](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/673bda40-5c31-4320-8294-c290fe39e353)

## Data duplicate:
![Screenshot 2024-03-12 101953](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/82a171f5-1959-4354-9853-6eb38197339a)

## Print Data:
![Screenshot 2024-03-12 094139](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/6f54d4e1-a3c7-489b-b504-5ba48939155e)
![Screenshot 2024-03-12 102852](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/46550dd1-70b7-43e0-9f45-45fb1ea790e2)
![Screenshot 2024-03-12 102549](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/8089016c-83a3-4fc9-90ee-58c052ddb775)
![Screenshot 2024-03-12 102559](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/92a48099-7a50-4f39-9051-dd7fff13607a)
![Screenshot 2024-03-12 102619](https://github.com/BalajiRajivel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103949835/db2796d2-05f8-4cda-8122-54de9b67c100)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
