import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="prabhusm/predictive_maintenance_project_model", filename="best_predictive_maintenance_project_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for predictive_maintenance_project
st.title("Predictive Maintenance Project App")
st.write("""
This application predicts the likelihood of a engine failure based on its parameters.
Please enter the data below to get a prediction.
""")

# User input
Engine_rpm = st.number_input("Engine rpm", min_value=50, max_value=3000, value=500)
Lub_oil_pressure = st.number_input("Lubricant oil pressure",min_value=0,max_value=10,value=3)
Fuel_pressure = st.number_input("Fuel pressure",min_value=0,max_value=30,value=6)
Coolant_pressure = st.number_input("Coolant pressure",min_value=0,max_value=10,value=2)
lub_oil_temp = st.number_input("Lubricant oil temperature", min_value=60, max_value=100,value=70)
Coolant_temp = st.number_input("Coolant temperature", min_value=60, max_value=300,value=70)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        'Engine rpm': Engine_rpm,
        'Lub oil pressure': Lub_oil_pressure,
        'Fuel pressure': Fuel_pressure,
        'Coolant pressure': Coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': Coolant_temp
       
    }])

    # Ensure string column names
    input_data.columns = input_data.columns.astype(str)

    st.write("Input Data")
    st.dataframe(input_data)

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Engine is likely to fail.")
    else:
        st.error("Engine is unlikely to fail!")
