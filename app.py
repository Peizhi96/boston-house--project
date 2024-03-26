import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler   
from flask import Flask, request, jsonify, render_template, url_for, app

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data= request.form['data']
    print(data)
    print(np.array(list(data.split(','))).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.split(','))).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == '__main__':
    app.run(debug=True)