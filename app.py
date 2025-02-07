import streamlit as st
import pickle
import numpy as np

# List of available models
models = {
    "Logistic Regression": "/home/momtahin/Desktop/Heart Disease Detaction/src/decision_tree_model.pkl",
    "Random Forest": "/home/momtahin/Desktop/Heart Disease Detaction/src/gradient_boosting_model.pkl",
    "Gradient Boosting": "/home/momtahin/Desktop/Heart Disease Detaction/src/k-nearest_neighbors_model.pkl",
    "Support Vector Machine": "/home/momtahin/Desktop/Heart Disease Detaction/src/logistic_regression_model.pkl.pkl",
    "K-Nearest Neighbors": "/home/momtahin/Desktop/Heart Disease Detaction/src/naive_bayes_model.pkl",
    "Decision Tree": "/home/momtahin/Desktop/Heart Disease Detaction/src/random_forest_model.pkl",
    "Naive Bayes": "/home/momtahin/Desktop/Heart Disease Detaction/src/support_vector_machine_model.pkl"
}

# Load selected model
selected_model = st.selectbox("Choose a model for prediction:", list(models.keys()))
with open(models[selected_model], 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open("/home/momtahin/Desktop/Heart Disease Detaction/src/scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# User input for required features
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert categorical inputs into numerical format
sex = 1 if sex == "Male" else 0
chest_pain = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}[chest_pain]
resting_ecg = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]

# Prepare input data for prediction
input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
input_data = scaler.transform(input_data)



prediction = model.predict(input_data)[0]


st.subheader('Prediction:')
st.write("Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected")


