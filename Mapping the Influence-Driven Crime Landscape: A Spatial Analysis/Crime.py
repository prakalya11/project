import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load model and data
def load_model_and_data(model_path='pgm.joblib', data_path='data_frame.joblib'):
    model = joblib.load(model_path)
    df = joblib.load(data_path)
    return model, df

# Store LabelEncoders for encoding and decoding
encoders = {}

# Function to encode categorical features
def encode_data(df, input_data):
    for column in ['entity', 'type_of_crime', 'subtype_of_crime', 'month', 'modality']:
        if column in df.columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                input_data[column] = le.transform(input_data[column].astype(str))
                encoders[column] = le  # Save the encoder for decoding
    return input_data

def main():
    st.title("Mapping the Influence-Driven Crime Landscape: A Spatial Analysis")

    # Load the model and DataFrame
    model, df = load_model_and_data()

    # Input fields
    year = st.number_input('Year', min_value=2000, max_value=2100, value=2024, step=1)
    entity = st.text_input('Entity', 'EntityA')
    type_of_crime = st.text_input('Type of Crime', 'CrimeType1')
    subtype_of_crime = st.text_input('Subtype_of_Crime', 'SubTypeA')
    month = st.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    modality = st.text_input('Modality', 'ModalityX')

    # Prepare a single row for prediction
    input_data = df.iloc[[0]].copy()  # copy the first row for structure

    # Update the row with user input
    input_data['year'] = year
    input_data['entity'] = entity
    input_data['type_of_crime'] = type_of_crime
    input_data['subtype_of_crime'] = subtype_of_crime
    input_data['month'] = month
    input_data['modality'] = modality

    # Encode the input data
    input_data = encode_data(df, input_data)

    # Button to predict
    if st.button('Predict'):
        predicted_code = model.predict(input_data)
        
        # Convert prediction to an integer if needed
        if isinstance(predicted_code, np.ndarray) and predicted_code.size == 1:
            predicted_code = predicted_code[0]

        # Decode the predicted type of crime, ensuring it is an integer index
        try:
            predicted_crime = encoders['type_of_crime'].inverse_transform([int(predicted_code)])[0]
            st.write(f"Predicted Type of Crime: {predicted_crime}")
        except (ValueError, IndexError) as e:
            st.error(f"Error in decoding the prediction: {e}")

if __name__ == "__main__":
    main()
