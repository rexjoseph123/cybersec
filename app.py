import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
dataset_path = "./Training_Dataset_Processed.csv"

try:
    # Load the dataset
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
    print(data.head())  # Display the first few rows for verification
except FileNotFoundError:
    print(f"Dataset not found at {dataset_path}")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
# Feature engineering
data['num_subdomains'] = data['having_Sub_Domain'].apply(lambda x: 1 if x >= 0 else 0)
data['security_score'] = (
        data['SSLfinal_State'] +
        data['Domain_registeration_length'] +
        data['HTTPS_token']
)

# Prepare the features and target
X = data.drop(columns=['Result'])
y = data['Result']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define Streamlit UI
st.title("Phishing Website Prediction")

# Accept a sample URL from the user
user_url = st.text_input("Enter a sample URL:")

if user_url:
    # Simulate feature extraction from the input URL (replace with actual feature extraction logic)
    st.subheader("Extracted Features")

    # Placeholder extracted features
    extracted_features = {
        "having_Sub_Domain": 1 if "sub" in user_url else -1,
        "SSLfinal_State": 1 if "https" in user_url else -1,
        "Domain_registeration_length": 1,
        "HTTPS_token": 1 if "https" in user_url else 0,
        "security_score": 0  # Placeholder; computed later
    }

    # Compute security_score based on extracted features
    extracted_features["security_score"] = (
            extracted_features["SSLfinal_State"] +
            extracted_features["Domain_registeration_length"] +
            extracted_features["HTTPS_token"]
    )

    # Display extracted features
    st.write(extracted_features)

    # Prepare the input data for prediction
    user_input = pd.DataFrame([list(extracted_features.values())],
                              columns=['having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
                                       'HTTPS_token', 'security_score'])

    # Add any missing columns with default values (e.g., zeros)
    for column in X.columns:
        if column not in user_input.columns:
            user_input[column] = 0  # Default value for missing columns

    # Ensure the columns match the model's expected input
    user_input = user_input[X.columns]

    # Make prediction
    prediction = model.predict(user_input)
    result = "Legitimate" if prediction == 1 else "Phishing"

    # Display the prediction
    st.subheader("Prediction")
    st.write(f"The URL is predicted to be: {result}")
