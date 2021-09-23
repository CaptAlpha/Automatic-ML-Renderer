import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import os
print(os.listdir())

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))
encoder = joblib.load(open('encoder.pkl','rb'))
scaler = joblib.load(open('scaler.pkl','rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''

    # data = request.get_json(force=True)
    data = "data.json"
    df = pd.read_json(data)
    for col in df:
        if(df[col].dtype=='object'):
            df[col]=encoder.transform(df[col])
    x=df.iloc[:,:].values

    x[:,:]=scaler.transform(x[:,:])

    prediction = model.predict(x)
    #Returning class labels
    prediction = encoder.inverse_transform(prediction)
    output = {"target" : {"1" : prediction[0]}}
    for i in range(1,len(prediction)):
        output["target"][str(i + 1)] = prediction[i]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)