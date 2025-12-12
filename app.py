import streamlit as st
import joblib
import pandas as pd
import numpy as np

scaler=joblib.load('standard_scaler.pkl')
le_gender=joblib.load('label_encoder_gender.pkl')
le_smoker=joblib.load('label_encoder_smoker.pkl')
le_diabetic=joblib.load('label_encoder_diabetic.pkl')
model =joblib.load('best_model.pkl')

st.set_page_config(page_title="Health Insurance Cost Prediction",layout="centered")
st.title("Health Insurance Cost Prediction")
st.write("Enter the details below to predict the health insurance cost.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=8, value=0)
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender=st.selectbox("Gender (F=0,M=1)", options=le_gender.classes_)
        diabetic=st.selectbox("Diabetic", options=le_diabetic.classes_)
        smoker=st.selectbox("Smoker", options=le_smoker.classes_)
    submitted = st.form_submit_button("Predict Payment")


if submitted:
    input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'bloodpressure': [bloodpressure],
    'diabetic': [diabetic]
    })

    input_data['gender'] =le_gender.transform(input_data['gender'])
    input_data['smoker'] =le_smoker.transform(input_data['smoker'])
    input_data['diabetic'] =le_diabetic.transform(input_data['diabetic'])

    num_cols=['age', 'bmi', 'children', 'bloodpressure']
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    prediction = model.predict(input_data)
    st.success(f"The predicted health insurance cost is: ${prediction[0]:.2f}")

