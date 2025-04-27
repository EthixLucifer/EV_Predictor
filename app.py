from flask import Flask, render_template, request
import ev_range_predictor
import numpy as np  # Assuming battery_model.py contains your model code

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')  # Renders an HTML template

@app.route('/predict', methods=['POST'])
def predict():
  # Extract user input from the form
  data = request.form
  # Convert user input to appropriate format (might require data cleaning)
  input_values = [float(data[key]) for key in data.keys() if key != 'submit']
  # Make prediction using the model
  prediction = ev_range_predictor.predict(np.array([input_values]))
  return render_template('result.html', prediction=prediction[0])  # Render result template

if __name__ == '__main__':
  app.run(debug=True)