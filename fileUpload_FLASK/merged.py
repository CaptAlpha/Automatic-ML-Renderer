from flask import Flask
from flask import render_template
from flask import request
import os
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    accuracy=0
    if request.method == 'POST':
        file = request.files['csvfile']
        tar=request.form['target']
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        df=pd.read_csv(filepath)

        # Identifying the categotical columns and label encoding them
        le = LabelEncoder()
        for col in df:
            if(df[col].dtype=='object'):
                df[col]=le.fit_transform(df[col])

        # Identifying the columns with null values and filling them with mean
        for col in df:
            if(df[col].isnull().sum()!=0):
                df[col]=df[col].fillna(df[col].dropna().mean())

        #After taking taregt as ip (modify this later)
        a=df.pop(tar)
        df[tar]=a
        x=df.iloc[:,:-1].values
        y=df.iloc[:,-1].values
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
        sc=StandardScaler()
        x_train[:,:]=sc.fit_transform(x_train[:,:])
        x_test[:,:]=sc.fit_transform(x_test[:,:])
        regressor = LinearRegression()
        regressor.fit(x_train,y_train)    
        from sklearn.metrics import r2_score
        y_pred=regressor.predict(x_test)
        accuracy=r2_score(y_test, y_pred)

    return render_template('index.html', prediction_text='Accuracy of the trained model is  {}'.format(accuracy))

if __name__ == '__main__':
    app.run(debug=True)