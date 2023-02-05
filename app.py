import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

model = pickle.load(open('irispredpkl.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods = ["POST"])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('index.html', data=pred)




if __name__ =="__main__":
    app.run(debug=True)