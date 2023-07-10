# app.py
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open(
    'logistic.pkl', 'rb'))
model2 = pickle.load(open(
    'decision.pkl', 'rb'))
model3 = pickle.load(
    open('naive.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    float_features = [
        float(request.form.get('pregnancies')),
        float(request.form.get('glucose')),
        float(request.form.get('blood_pressure')),
        float(request.form.get('skin_thickness')),
        float(request.form.get('insulin')),
        float(request.form.get('bmi')),
        float(request.form.get('dpf')),
        float(request.form.get('age'))
    ]

    selected_model = request.form.get('model')

    if selected_model == 'logistic':
        prediction = model1.predict([float_features])
    elif selected_model == 'decision':
        prediction = model2.predict([float_features])
    elif selected_model == 'naive':
        prediction = model3.predict([float_features])
    else:
        return render_template('index.html', prediction_text='Invalid model selection.')

    text = "No Possible Risk"
    if prediction == 1:
        text = "Possible Risk"
    return render_template('index.html', prediction_text='Outcome: {}'.format(text))


if __name__ == "__main__":
    app.run(debug=True)
