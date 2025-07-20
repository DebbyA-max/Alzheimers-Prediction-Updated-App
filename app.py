
import streamlit as st
import numpy as np
import joblib

model = joblib.load("alzheimers_model_2.pkl")
st.title("Alzheimer's Disease Prediction App")
st.write("Enter the patient data below to predict Alzheimer's Diagnosis.")


FunctionalAssessment = st.number_input("Functional Assessment Score", min_value=0.0, max_value=10.0, value=0.0, key="fun_acc_input")
ADL = st.number_input("Activities of Daily Living Score", min_value=0.0, max_value=10.0, value=0.0, key="ADL_input")
MMSE = st.number_input("Mini-Mental State Examination score", min_value=0.0, max_value=30.0, value=0.0, key="MMSE_input")

MemoryComplaints = st.radio("Presence of Memory Complaints", ["Yes", "No"], key="mce_input")
MemoryComplaints_encoded = 1 if MemoryComplaints == "Yes" else 0

PhysicalActivity = st.number_input("Weekly Physical Activity (Hours)", min_value=0.0, max_value=10.0, value=0.0, key="pha_input")
SleepQuality = st.number_input("Sleep Quality Score", min_value=4.0, max_value=10.0, value=4.0, key="slp_input")
CholesterolHDL = st.number_input("High Density Lipoprotein Cholesterol Level (mg/dL)", min_value=20.0, max_value=100.0, value=20.0, key="chdl_input")
CholesterolTriglycerides = st.number_input("Triglycerides Level (mg/dL)", min_value=50.0, max_value=400.0, value=50.0, key="ctry_input")
AlcoholConsumption = st.number_input("Weekly Alcohol Consumption (Units)", min_value=0.0, max_value=20.0, value=0.0, key="alc_input")
DietQuality = st.number_input("Diet Quality Score", min_value=0.0, value=0.0, key="diq_input")
CholesterolLDL = st.number_input("Low Density Lipoprotein Cholesterol Level (mg/dL)", min_value=50.0, max_value=200.0, value=50.0, key="cldl_input")
SystolicBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=90.0, max_value=180.0, value=90.0, key="sbp_input")
BMI = st.number_input("Body Mass Index", min_value=15.0, max_value=40.0, value=15.0, key="bmi_input")
CholesterolTotal = st.number_input("Total Cholesterol Level (mg/dL)", min_value=150.0, max_value=300.0, value=150.0, key="cht_input")

BehavioralProblems = st.radio("Presence of Behavioral Problems", ["Yes", "No"], key="bpe_input")
BehavioralProblems_encoded = 1 if BehavioralProblems == "Yes" else 0

DiastolicBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=60.0, max_value=120.0, value=60.0, key="dbp_input")
Age = st.number_input("Patient's Age", min_value=60.0, max_value=90.0, value=60.0, key="age_input")

education_options = [
    "0: None",
    "1: High School",
    "2: Bachelor's",
    "3: Higher"
]
education_selected = st.radio("Select the Patient's Education Level:", education_options, key="edu_input")
EducationLevel_encoded = float(education_selected.split(":")[0])  # float to match others

input_data = np.array([[
    Age, EducationLevel_encoded, DietQuality, SleepQuality, PhysicalActivity, 
    ADL, AlcoholConsumption, BMI, DiastolicBP, SystolicBP, 
    CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, CholesterolTotal,  
    MemoryComplaints_encoded, BehavioralProblems_encoded, FunctionalAssessment, MMSE,
]])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {prediction}")
