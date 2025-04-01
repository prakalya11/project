import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load pre-trained models and scalers for both anomaly detection and threat prediction
anomaly_model = joblib.load('isolation_forest_model.joblib')
scaler_anomaly = joblib.load('scaler1.joblib')

threat_model = joblib.load('logistic_regression_model.joblib')
scaler_threat = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')


# Preprocessing function for anomaly detection
def preprocess_anomaly_data(dataframe):
    dataframe = dataframe.select_dtypes(include=[np.number])

    # Handle missing values by filling with the mode of each column
    dataframe = dataframe.fillna(dataframe.mode().iloc[0])  # Fill missing values with mode

    # Handle infinite values by replacing them with NaN, and then filling NaNs
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe = dataframe.fillna(dataframe.mode().iloc[0])  # Replace NaN with the mode of the column

    # Clip extreme values to avoid overflow errors or too large values
    dataframe = dataframe.applymap(lambda x: min(max(x, -1e10), 1e10))  # Clip values to a reasonable range

    # Scale the data using the pre-trained scaler
    X_scaled = scaler_anomaly.transform(dataframe)

    return X_scaled


# Preprocessing function for threat prediction
def preprocess_threat_data(dataframe):
    dataframe = dataframe.fillna(dataframe.mode().iloc[0])  # Handle missing values by filling with mode
    categorical_columns = ['Attack_Severity', 'Botnet_Family', 'Malware_Type', 'System_Patch_Status']

    for column in categorical_columns:
        if column in dataframe.columns:
            unique_values = dataframe[column].unique()
            unseen_values = [value for value in unique_values if value not in label_encoder.classes_]
            if unseen_values:
                dataframe[column] = dataframe[column].replace(unseen_values, 'Unknown')
                if 'Unknown' not in label_encoder.classes_:
                    label_encoder.classes_ = np.append(label_encoder.classes_, ['Unknown'])

            dataframe[column] = label_encoder.transform(dataframe[column])

    for col in dataframe.select_dtypes(include=[object]).columns:
        if dataframe[col].str.contains('\d+\.\d+\.\d+\.\d+', regex=True).any():
            dataframe = dataframe.drop(columns=[col])  # Drop columns containing IP addresses

    dataframe = dataframe.drop(['Label', 'Threat_Severity', 'Date'], axis=1, errors='ignore')  # Drop irrelevant columns
    dataframe = dataframe.clip(-1e10, 1e10)  # Clip extreme values if necessary
    scaled_data = scaler_threat.transform(dataframe)
    return scaled_data


# Anomaly Detection Logic
def anomaly_detection(df):
    df_processed = preprocess_anomaly_data(df)
    y_pred = anomaly_model.predict(df_processed)
    df['Anomaly'] = y_pred
    df['Anomaly'] = df['Anomaly'].map({-1: 'Anomaly', 1: 'Normal'})  # Map prediction to labels
    st.write("### Anomalies Detected:")
    st.write(df[df['Anomaly'] == 'Anomaly'])  # Show anomalies
    #fig, ax = plt.subplots()
    #sns.scatterplot(x=df.index, y=df['Anomaly'].apply(lambda x: 1 if x == 'Anomaly' else 0), ax=ax)
    #st.pyplot(fig)
    csv = df.to_csv(index=False)
    st.download_button("Download results as CSV", csv, "anomaly_detection_results.csv")


# Threat Prediction Logic
def threat_prediction(df):
    processed_input_data = preprocess_threat_data(df)
    threat_level_mapping = {
        0: "No threat",
        1: "Low-level threat",
        2: "Medium-level threat",
        3: "High-level threat",
        
        4: "Critical threat"
    }

    prediction_mode = st.radio("Select prediction mode", ("Model Prediction"))
    sample_indices = list(range(1, len(df) + 1))
    selected_sample_index = st.selectbox("Select a sample to display prediction", sample_indices)

    if prediction_mode == "Model Prediction":
        prediction = threat_model.predict(processed_input_data[selected_sample_index - 1:selected_sample_index])
        predicted_severity = prediction[0]
        threat_level = threat_level_mapping.get(predicted_severity, "Unknown threat level")
        st.write(f"Prediction for Sample {selected_sample_index} (Model): {threat_level}")
        st.write("""
    - **No Threat (Severity = 0)**: No action needed unless additional data triggers re-evaluation.
    - **Low-Level Threat (Severity = 1)**: Lower priority, handled as resources allow.
    - **Medium-Level Threat (Severity = 2)**: Standard response.
    - **High-Level Threat (Severity = 3)**: High priority response within a set timeframe.
    - **Critical Threat (Severity = 4)**: Immediate response.
""")


# Streamlit App Logic
def run():
    # Streamlit UI
    st.title("Security Breach Analysis")
    st.write("Choose an analysis type: Anomaly Detection or Threat Prediction.")

    # Choose analysis type
    analysis_type = st.radio("Select the analysis type", ("Anomaly Detection", "Threat Prediction"))

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Overview:")
        st.write(df.head())  # Show a preview of the uploaded file

        if analysis_type == "Anomaly Detection":
            anomaly_detection(df)

        elif analysis_type == "Threat Prediction":
            threat_prediction(df)


if __name__ == "__main__":
    run()
