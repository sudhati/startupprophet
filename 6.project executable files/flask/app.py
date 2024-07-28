# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open("Randf.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contactus')
def contactus():
    return render_template("contactus.html")

@app.route('/info')
def info():
    return render_template("info.html")

@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template("predict.html")

@app.route('/submit', methods=["POST"])
def submit():
    # Extract input features from the form and convert to float
    org_name = request.form.get('org_name')
    input_feature = [float(x) for x in request.form.values() if x != org_name]
    x = [np.array(input_feature)]

    # Create a DataFrame with the input features
    names = ['is_ecommerce', 'is_otherstate', 'has_VC', 'has_angel',
             'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD',
             'is_top500', 'relationships', 'funding_rounds', 'milestones']
    data = pd.DataFrame(x, columns=names)

    # Make prediction using the loaded model
    pred = model.predict(data)

    # Determine the prediction result based on the model's output
    if pred == 1:
        prediction_result = "Success"
    else:
        prediction_result = "Failure"

    return render_template('submit.html', pred=prediction_result, org_name=org_name)

if __name__ == "__main__":
    app.run(port=5000)
