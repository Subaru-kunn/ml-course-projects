from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load the scaler and models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('naive_bayes_model.pkl', 'rb') as f:
    loaded_nb_model = pickle.load(f)
with open('perceptron_model.pkl', 'rb') as f:
    loaded_perceptron_model = pickle.load(f)

def preprocess_input(data):
    try:
        # Convert input data to float
        input_features = np.array([[float(data["age"]), float(data["glucose"]), float(data["insulin"]), float(data["bmi"])]])
        # Apply scaling and clip values to prevent outliers
        input_features = scaler.transform(input_features)
        input_features = np.clip(input_features, -3, 3)  # Clip values within 3 standard deviations
    except ValueError:
        raise ValueError("Invalid input type. Please ensure all inputs are numeric.")
    return input_features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'naive_bayes')

    try:
        input_features = preprocess_input(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Select the model and make a prediction
    if model_type == 'naive_bayes':
        prediction = loaded_nb_model.predict(input_features)
    elif model_type == 'perceptron':
        prediction = loaded_perceptron_model.predict(input_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    return jsonify({'diabetes_type': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
