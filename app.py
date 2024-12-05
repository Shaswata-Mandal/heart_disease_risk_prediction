# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import gzip

# Load the trained model
model_path = 'model.pkl.gz'
with gzip.open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Heart Disease Risk' if prediction[0] == 1 else 'No Heart Disease Risk'

    #Check if a warning popup is needed
    show_warning= prediction[0]==1

    return render_template('index.html', prediction_text='Prediction: {}'.format(output), show_warning=show_warning)

if __name__ == "__main__":
    app.run(debug=True)

    