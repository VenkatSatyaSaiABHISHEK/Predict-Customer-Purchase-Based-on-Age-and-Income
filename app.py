from flask import Flask, render_template, request
from joblib import load
import numpy as np


app = Flask(__name__)
model = load('model/linear_model_customer.joblib')

# Landing page
@app.route('/')
def index():
    return render_template('index.html')  # This is your landing page

# Form page
@app.route('/form')
def form():
    return render_template('form.html')  # This is your prediction input form

# Prediction result
@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    income = float(request.form['income'])
    prediction = model.predict(np.array([[age, income]]))[0]
    result = "Will Buy the Product" if prediction >= 0.5 else "Will Not Buy the Product"
    return render_template('result.html', age=age, income=income, result=result)

if __name__ == '__main__':
    app.run(debug=True)
