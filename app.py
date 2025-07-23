
import streamlit as st
import numpy as np
import joblib

model = joblib.load("alzheimers_model_2.pkl")
st.title("Alzheimer's Disease Prediction App")
st.write("Enter the patient's data below to predict Alzheimer's Diagnosis.")

Age = st.number_input("Patient's Age (60-90 years)", min_value=60, max_value=90, value=60, step=1, format="%d", key="age_input")

education_options = [
    "0: None",
    "1: High School",
    "2: Bachelor's",
    "3: Higher"
]
education_selected = st.radio("Select the Patient's Education Level:", education_options, key="edu_input")
EducationLevel_encoded = int(education_selected.split(":")[0])

DietQuality = st.number_input("Diet Quality Score (0-10)", min_value=0.0,  max_value=10.0, value=0.0, step=0.1, format="%.2f", key="diq_input")
SleepQuality = st.number_input("Sleep Quality Score (4-10)", min_value=4.0, max_value=10.0, value=4.0, step=0.1, format="%.2f", key="slp_input")
PhysicalActivity = st.number_input("Weekly Physical Activity (0-10 hours)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.2f", key="pha_input")
ADL = st.number_input("Activities of Daily Living Score (0-10)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.2f", key="ADL_input")
AlcoholConsumption = st.number_input("Weekly Alcohol Consumption (0-20 Units)", min_value=0.0, max_value=20.0, value=0.0, step=0.1, format="%.2f", key="alc_input")
BMI = st.number_input("Body Mass Index (15-40)", min_value=15.0, max_value=40.0, value=15.0, step=0.1, format="%.2f", key="bmi_input")
SystolicBP = st.number_input("Systolic Blood Pressure (90-180 mmHg)", min_value=90, max_value=180, value=90, step=1, format="%d", key="sbp_input")
DiastolicBP = st.number_input("Diastolic Blood Pressure (60-120 mmHg)", min_value=60, max_value=120, value=60, step=1, format="%d", key="dbp_input")
CholesterolLDL = st.number_input("Low Density Lipoprotein Cholesterol Level (50-200 mg/dL)", min_value=50.0, max_value=200.0, value=50.0, step=0.1, format="%.2f", key="cldl_input")
CholesterolHDL = st.number_input("High Density Lipoprotein Cholesterol Level (20-100 mg/dL)", min_value=20.0, max_value=100.0, value=20.0, step=0.1, format="%.2f", key="chdl_input")
CholesterolTriglycerides = st.number_input("Triglycerides Level (50-400 mg/dL)", min_value=50.0, max_value=400.0, value=50.0, step=0.1, format="%.2f", key="ctry_input")
CholesterolTotal = st.number_input("Total Cholesterol Level (150-300 mg/dL)", min_value=150.0, max_value=300.0, value=150.0, step=0.1, format="%.2f", key="cht_input")

MemoryComplaints = st.radio("Presence of Memory Complaints", ["Yes", "No"], key="mce_input")
MemoryComplaints_encoded = 1 if MemoryComplaints == "Yes" else 0

BehavioralProblems = st.radio("Presence of Behavioral Problems", ["Yes", "No"], key="bpe_input")
BehavioralProblems_encoded = 1 if BehavioralProblems == "Yes" else 0

FunctionalAssessment = st.number_input("Functional Assessment Score (0-10)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.2f", key="fun_acc_input")
MMSE = st.number_input("Mini-Mental State Examination score (0-30)", min_value=0.0, max_value=30.0, value=0.0, step=0.1, format="%.2f", key="MMSE_input")


input_data = np.array([[
    Age, EducationLevel_encoded, DietQuality, SleepQuality, PhysicalActivity, 
    ADL, AlcoholConsumption, BMI,  SystolicBP, DiastolicBP,
    CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, CholesterolTotal,  
    MemoryComplaints_encoded, BehavioralProblems_encoded, FunctionalAssessment, MMSE,
]])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(f"Prediction: {prediction}, Patient may have Alzheimer's disease.")
    else:
        st.success(f"Prediction: {prediction}, Patient is unlikely to have Alzheimer's disease.")
