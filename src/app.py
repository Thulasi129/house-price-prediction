

import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model and scaler
model = joblib.load('results/model.joblib')
scaler = joblib.load('results/scaler.joblib')
model_columns = joblib.load('results/model_columns.joblib')

# Set up the Streamlit app title
st.title('House Price Prediction')

# Create input fields for user to enter house details
st.sidebar.header('Input House Details')
area = st.sidebar.number_input('Area (sq. ft.)', min_value=500, max_value=5000, value=1500)
bedrooms = st.sidebar.selectbox('Bedrooms', [1, 2, 3, 4, 5], index=2)
age = st.sidebar.number_input('Age (years)', min_value=1, max_value=100, value=10)
location = st.sidebar.selectbox('Location', ['suburban', 'rural', 'urban'], index=0)

# Create a 'Predict' button
if st.sidebar.button('Predict'):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'age': [age],
        'location': [location]
    })

    # One-hot encode the 'location' feature
    input_data = pd.get_dummies(input_data, columns=['location'], drop_first=True, dtype=int)

    # Align the input data columns with the model's expected columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale the numerical features
    numerical_features = ['area', 'bedrooms', 'age']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Make a prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write('## Predicted Price')
    st.write(f'The estimated price of the house is: ${prediction[0]:,.2f}')
