from flask import Flask
from flask import render_template
from flask import request
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csvfile']
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        return file.filename

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)