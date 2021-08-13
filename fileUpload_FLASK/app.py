from flask import Flask
from flask import render_template
from flask import request
from flask import send_file
from flask import current_app
from flask import session
from flask import redirect
from flask import send_from_directory
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    accuracy=0
    final=''
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

        #Training the model
        regressor = LinearRegression()
        regressor.fit(x_train,y_train)    
        from sklearn.metrics import r2_score
        y_pred=regressor.predict(x_test)
        accuracy=r2_score(y_test, y_pred)
        

        final="""
!pip install numpy
!pip install pandas
!pip install scikit-learn
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

filepath=   #Please enter the filepath of csv file.
tar=    #Please enter target variable name
df=pd.read_csv(filepath)

# Identifying the categotical columns and label encoding them
le = LabelEncoder()
for col in df:
    if(df[col].dtype=='object'):
        df[col]=le.fit_transform(df[col])

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

#Training the model
regressor = LinearRegression()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)
"""

        code=open("static/output.py","w")
        code.write(final)

    return render_template('index.html', prediction_text='Trained Model with accuracy {}'.format(accuracy))
"""
@app.route('/return-files/')
def return_files_tut():
	try:
		return send_file('static/output.py', attachment_filename='output.py')
	except Exception as e:
		return str(e)
"""

@app.route('/return-files/', methods=['GET', 'POST'])
def download(filename):
    # Appending app path to upload folder path within app root folder
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    # Returning file from appended path
    return send_from_directory(directory=uploads, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
