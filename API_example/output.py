import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

filepath= "Iris.csv"   #Please enter the filepath of csv file.
tar= "Species"   #Please enter target variable name
df=pd.read_csv(filepath)

# Identifying the categotical columns and label encoding them
le = LabelEncoder()
for col in df:
    if(df[col].dtype=='object'):
        df[col]=le.fit_transform(df[col])
joblib.dump(le,"encoder.pkl")

# Identifying the columns with null values and filling them with mean
for col in df:
    if(df[col].isnull().sum()!=0):
        df[col]=df[col].fillna(df[col].dropna().median())

#Train-Test Split
a=df.pop(tar)
df[tar]=a
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#Scaling
sc=StandardScaler()
x_train[:,:]=sc.fit_transform(x_train[:,:])
x_test[:,:]=sc.fit_transform(x_test[:,:])
joblib.dump(sc,"scaler.pkl")

#Training the model
from sklearn.linear_model import LinearRegression,LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(x_train,y_train)    
from sklearn.metrics import accuracy_score
y_pred=classifier.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
joblib.dump(classifier, 'model.pkl')