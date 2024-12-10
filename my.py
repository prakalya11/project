import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('work.joblib')


# Assuming you have a label encoder used for encoding the inputs
label_enc = LabelEncoder()

# Streamlit App
st.title('Preference Prediction')

# Get user input
description = st.text_input('Enter Product :')
quantity = st.number_input('Enter Quantity:', min_value=1)

if st.button('Predict'):
    if description and quantity:
        # Prepare the input data
        input_data = pd.DataFrame([[description, quantity]], columns=['Description', 'Quantity'])

        # Convert inputs to string and encode them as required
        input_data['Description'] = label_enc.fit_transform(input_data['Description'].astype(str))
        input_data['Quantity'] = input_data['Quantity'].astype(str)
        input_data['Quantity'] = label_enc.fit_transform(input_data['Quantity'])

        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction
        st.write(f'Predicted Country: {prediction}')
    else:
        st.write('Please provide both description and quantity.')

