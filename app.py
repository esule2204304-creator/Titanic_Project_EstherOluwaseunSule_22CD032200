"""
Titanic Survival Prediction Web Application
Student: Esther Oluwaseun Sule
Matric No: 22CD032200
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model artifacts
MODEL_PATH = 'model/titanic_survival_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
FEATURES_PATH = 'model/feature_names.pkl'

print("="*70)
print("TITANIC SURVIVAL PREDICTION WEB APPLICATION")
print("Student: Esther Oluwaseun Sule")
print("Matric No: 22CD032200")
print("="*70)
print("\nLoading model artifacts...")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✓ Model loaded successfully!")
    print(f"✓ Features: {feature_names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Feature mappings
EMBARKED_MAP = {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}
SEX_MAP = {0: 'Female', 1: 'Male'}
CLASS_MAP = {1: 'First Class', 2: 'Second Class', 3: 'Third Class'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])
        
        if pclass not in [1, 2, 3]:
            return jsonify({'error': 'Invalid passenger class'}), 400
        if sex not in [0, 1]:
            return jsonify({'error': 'Invalid sex value'}), 400
        if age < 0 or age > 120:
            return jsonify({'error': 'Invalid age'}), 400
        if fare < 0:
            return jsonify({'error': 'Invalid fare'}), 400
        if embarked not in [0, 1, 2]:
            return jsonify({'error': 'Invalid embarkation port'}), 400
        
        input_data = pd.DataFrame([[pclass, sex, age, fare, embarked]], columns=feature_names)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = {
            'prediction': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survival_probability': float(probability[1] * 100),
            'death_probability': float(probability[0] * 100),
            'confidence': float(max(probability) * 100),
            'passenger_info': {
                'Passenger Class': f"{CLASS_MAP[pclass]} ({pclass})",
                'Sex': SEX_MAP[sex],
                'Age': f"{age} years",
                'Fare': f"${fare:.2f}",
                'Embarked From': EMBARKED_MAP[embarked]
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'student': 'Esther Oluwaseun Sule'})

if __name__ == '__main__':
    app.run()

# For Vercel
app = app