"""
Titanic Survival Prediction - Model Development Script
Student: Esther Oluwaseun Sule
Matric No: 22CD032200

This script trains a Random Forest Classifier on the Titanic dataset
and saves the model artifacts for deployment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TITANIC SURVIVAL PREDICTION - MODEL DEVELOPMENT")
print("Student: Esther Oluwaseun Sule")
print("Matric No: 22CD032200")
print("="*70)

# Load the Titanic dataset
print("\n[1] Loading Titanic dataset...")
try:
    df = pd.read_csv('../data/train.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Survival rate: {df['Survived'].mean()*100:.2f}%")
except FileNotFoundError:
    print("❌ Error: train.csv not found in data folder!")
    print("Please ensure train.csv is in the data/ directory")
    exit(1)

# Display dataset info
print("\n[2] Dataset Overview:")
print("\nColumn Names:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum())

# Feature Selection - Using 5 features as required
print("\n[3] Selecting features...")
selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

df_model = df[selected_features + [target]].copy()
print(f"✓ Selected 5 features: {selected_features}")
print(f"✓ Target variable: {target}")

# Data Preprocessing
print("\n[4] Data Preprocessing:")

# Handle missing values
print("\n  a) Handling missing values...")
initial_missing = df_model.isnull().sum().sum()
print(f"     Missing values before: {initial_missing}")

# Age: Fill with median
df_model['Age'].fillna(df_model['Age'].median(), inplace=True)

# Embarked: Fill with mode (most common)
df_model['Embarked'].fillna(df_model['Embarked'].mode()[0], inplace=True)

# Fare: Fill with median
df_model['Fare'].fillna(df_model['Fare'].median(), inplace=True)

# Drop any remaining missing values
df_model.dropna(inplace=True)

print(f"     Missing values after: {df_model.isnull().sum().sum()}")
print(f"     Final dataset size: {df_model.shape[0]} samples")

# Encode categorical variables
print("\n  b) Encoding categorical variables...")
# Sex: male=1, female=0
df_model['Sex'] = df_model['Sex'].map({'male': 1, 'female': 0})

# Embarked: C=0, Q=1, S=2
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
df_model['Embarked'] = df_model['Embarked'].map(embarked_mapping)

print("     ✓ Sex encoded: male=1, female=0")
print("     ✓ Embarked encoded: C=0, Q=1, S=2")

# Separate features and target
X = df_model[selected_features]
y = df_model[target]

print(f"\n  c) Feature scaling...")
# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("     ✓ Features scaled using StandardScaler")

# Train-Test Split
print("\n[5] Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples ({(len(X_train)/len(X_scaled))*100:.0f}%)")
print(f"✓ Testing set: {X_test.shape[0]} samples ({(len(X_test)/len(X_scaled))*100:.0f}%)")

# Model Training - Random Forest Classifier
print("\n[6] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Model training completed!")

# Model Evaluation
print("\n[7] Evaluating model performance...")
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
print("="*70)

print("\nCLASSIFICATION REPORT - TEST SET")
print("="*70)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Did Not Survive', 'Survived']))
print("="*70)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print("\nInterpretation:")
print(f"  True Negatives (Correctly predicted deaths): {cm[0][0]}")
print(f"  False Positives (Incorrectly predicted survivals): {cm[0][1]}")
print(f"  False Negatives (Incorrectly predicted deaths): {cm[1][0]}")
print(f"  True Positives (Correctly predicted survivals): {cm[1][1]}")

# Feature Importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))
print("="*70)

# Save Model Artifacts
print("\n[8] Saving model artifacts...")
try:
    joblib.dump(rf_model, 'titanic_survival_model.pkl')
    print("✓ Model saved: titanic_survival_model.pkl")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Scaler saved: scaler.pkl")
    
    joblib.dump(selected_features, 'feature_names.pkl')
    print("✓ Feature names saved: feature_names.pkl")
    
    print("\n✓ All model artifacts saved successfully!")
except Exception as e:
    print(f"❌ Error saving model: {e}")
    exit(1)

# Demonstrate Model Reloading
print("\n[9] Demonstrating model reload (without retraining)...")
try:
    loaded_model = joblib.load('titanic_survival_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_features = joblib.load('feature_names.pkl')
    print("✓ Model artifacts successfully reloaded!")
    print(f"   Model type: {type(loaded_model).__name__}")
    print(f"   Features: {loaded_features}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Test Predictions with Sample Data
print("\n[10] Testing predictions with sample passengers...")
print("="*70)

# Sample 1: Male, 3rd class, age 22, fare 7.25, Southampton
sample_1 = pd.DataFrame([[3, 1, 22, 7.25, 2]], columns=loaded_features)
sample_1_scaled = loaded_scaler.transform(sample_1)
pred_1 = loaded_model.predict(sample_1_scaled)[0]
prob_1 = loaded_model.predict_proba(sample_1_scaled)[0]

print("\n📋 Sample Passenger 1:")
print("   Pclass: 3 (Third Class)")
print("   Sex: Male")
print("   Age: 22 years")
print("   Fare: $7.25")
print("   Embarked: Southampton")
print(f"   🔮 Prediction: {'Survived ✓' if pred_1 == 1 else 'Did Not Survive ✗'}")
print(f"   📊 Confidence: {max(prob_1)*100:.2f}%")

# Sample 2: Female, 1st class, age 38, fare 71.28, Cherbourg
sample_2 = pd.DataFrame([[1, 0, 38, 71.28, 0]], columns=loaded_features)
sample_2_scaled = loaded_scaler.transform(sample_2)
pred_2 = loaded_model.predict(sample_2_scaled)[0]
prob_2 = loaded_model.predict_proba(sample_2_scaled)[0]

print("\n📋 Sample Passenger 2:")
print("   Pclass: 1 (First Class)")
print("   Sex: Female")
print("   Age: 38 years")
print("   Fare: $71.28")
print("   Embarked: Cherbourg")
print(f"   🔮 Prediction: {'Survived ✓' if pred_2 == 1 else 'Did Not Survive ✗'}")
print(f"   📊 Confidence: {max(prob_2)*100:.2f}%")

# Sample 3: Female child, 2nd class, age 5, fare 21.00, Southampton
sample_3 = pd.DataFrame([[2, 0, 5, 21.00, 2]], columns=loaded_features)
sample_3_scaled = loaded_scaler.transform(sample_3)
pred_3 = loaded_model.predict(sample_3_scaled)[0]
prob_3 = loaded_model.predict_proba(sample_3_scaled)[0]

print("\n📋 Sample Passenger 3:")
print("   Pclass: 2 (Second Class)")
print("   Sex: Female (Child)")
print("   Age: 5 years")
print("   Fare: $21.00")
print("   Embarked: Southampton")
print(f"   🔮 Prediction: {'Survived ✓' if pred_3 == 1 else 'Did Not Survive ✗'}")
print(f"   📊 Confidence: {max(prob_3)*100:.2f}%")

print("\n" + "="*70)
print("✅ MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📦 Model artifacts ready for web application deployment:")
print("   • titanic_survival_model.pkl")
print("   • scaler.pkl")
print("   • feature_names.pkl")
print("\n🎓 Developed by: Esther Oluwaseun Sule (22CD032200)")
print("="*70)