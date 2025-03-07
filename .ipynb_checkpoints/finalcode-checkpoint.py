import streamlit as st
st.title('CROP YIELD PREDICTION')
st.header('prediction')
st.write('Predicting Crop Yield By Using The Values of Rainfall (mm),Soil Quality (index),Farm Size (hectares),Sunlight (hours),Fertilizer (kg).This prediction helps the farms to improve agricultural plannings')
from PIL import Image
st.image(Image.open(r'C:\Users\DELL\Downloads\crop yield image.webp'))

st.markdown(
    """
    <style>
    body  {
        font-family: 'Arial',sans-serif; /* Example font */
        background-color: #f4f4f4; /* Light background color */
        color: #333; /* Dark text color */
    }
    .main .block-container {
        max-width: 800px; /* Adjust container width */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #007bff; /* Example heading color (blue) */
        text-align: center; /* Center-align headings */
    }
    .stButton>button {
        background-color: #007bff; /* Blue button */
        color: white;
    }
    .stNumberInput, .stSlider  {
        margin-bottom: 1rem; /* Space between input elements */
    }
    .stSuccess {
        color: green;
        font-weight: bold;
    }
    .stError {
        color: red;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


import numpy as np
import joblib
model=joblib.load('finalcode.pkl')
scaler = joblib.load('scaler.pkl')

# User inputs
rainfall = st.number_input('Enter Rainfall (mm):', min_value=0, step=1)
soil_quality = st.slider('Soil Quality Index (1-10):', min_value=1, max_value=10, step=1)
farm_size = st.number_input('Enter Farm Size (hectares):', min_value=1, step=1)
sunlight = st.number_input('Enter Sunlight Hours:', min_value=1,max_value=12, step=1)
fertilizer = st.number_input('Enter Fertilizer Used (kg):', min_value=0, step=1)


if st.button('Predict'):
    try:
        # Scale input values
        input_data = np.array([[rainfall, soil_quality, farm_size, sunlight, fertilizer]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        st.success(f'Predicted Crop Yield: {prediction:.2f} ')
    except Exception as e:
        st.error(f'Error in prediction: {e}')


st.markdown("---")  # Separator line
st.markdown("<p style='text-align: center; font-size: small;'>Crop Yield Prediction App</p>", unsafe_allow_html=True)
