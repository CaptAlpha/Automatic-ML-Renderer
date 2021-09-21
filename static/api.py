import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))
encoder = joblib.load(open('encoder.pkl','rb'))
scaler = joblib.load(open('scaler.pkl','rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    '''
    For direct API calls trought request
    '''

    data = request.get_json(force=True)
    df = pd.read_json(data)

    for col in df:
        if(df[col].dtype=='object'):
            df[col]=encoder.transform(df[col])

    x=df.iloc[:,:].values
    x[:,:]=scaler.transform(x[:,:])

    prediction = model.predict(x)
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)