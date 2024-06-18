import pickle
from flask import Flask, request, jsonify
import numpy as np
from scripts.preprocess import preprocess_text

# Load the model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Flask app
app = Flask(__name__)

# Paths to log files
unknown_skills_log_path = 'unknown_skills.log'
low_confidence_log_path = 'low_confidence_predictions.log'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    skill = data['skill']
    processed_skill = preprocess_text(skill)
    prediction = model.predict([processed_skill])[0]
    prediction_prob = model.predict_proba([processed_skill])[0]
    
    # Get the confidence of the prediction
    confidence = np.max(prediction_prob)

    # Log low confidence predictions
    if confidence < 0.95:
        with open(low_confidence_log_path, 'a') as log_file:
            log_file.write(f"Skill: {skill}, Prediction: {prediction}, Confidence: {confidence:.2f}\n")
    
    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
