# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from make_pickle_object import make_pickle_object

# Load the Random Forest CLassifier model
make_pickle_object()
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = None
with open('diabetes-prediction-rfc-model.pkl', 'rb') as f:
    classifier = pickle.load(f) 

app = Flask(__name__,static_url_path="/static")

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True, port=12345)