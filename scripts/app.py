import pickle
from flask import Flask, request, jsonify
from preprocess import preprocess_text

# Load the model
with open('../models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Flask app
app = Flask(__name__)

# Path to log unknown skills
unknown_skills_log_path = 'unknown_skills.log'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    skill = data['skill']
    processed_skill = preprocess_text(skill)
    prediction = model.predict([processed_skill])
    
    # Log unknown skills if prediction is not in known labels
    if prediction[0] not in model.classes_:
        with open(unknown_skills_log_path, 'a') as log_file:
            log_file.write(f"{skill}\n")
        return jsonify({'prediction': 'unknown'})
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
