import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load or generate dataset
def load_data():
    try:
        df = pd.read_csv("credit_data.csv")  # Replace with actual file
    except FileNotFoundError:
        df = generate_synthetic_data()
    return df

# Generate synthetic dataset
def generate_synthetic_data():
    np.random.seed(42)
    data = {
        "Name": ["User" + str(i) for i in range(1000)],
        "Age": np.random.randint(21, 60, 1000),
        "Occupation": np.random.choice(["Salaried", "Self-Employed", "Unemployed"], 1000),
        "Income": np.random.randint(20000, 150000, 1000),
        "Banking_History": np.random.choice(["Good", "Average", "Poor"], 1000),
        "Loans": np.random.randint(0, 5, 1000),
        "Credit_Score": np.random.choice(["Poor", "Standard", "Good"], 1000)
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    X = df.drop(columns=["Name", "Credit_Score"])
    y = df["Credit_Score"]
    
    categorical_cols = ["Occupation", "Banking_History"]
    encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}
    for col, encoder in encoders.items():
        X[col] = encoder.transform(X[col])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled, encoders, scaler

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Load data
st.title("Credit Score Prediction App")
df = load_data()
X, y, encoders, scaler = preprocess_data(df)
model = train_model(X, y)

# User input form
st.sidebar.header("Enter Details")
age = st.sidebar.slider("Age", 21, 60, 30)
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Self-Employed", "Unemployed"])
income = st.sidebar.number_input("Income", 20000, 150000, 50000)
banking_history = st.sidebar.selectbox("Banking History", ["Good", "Average", "Poor"])
loans = st.sidebar.slider("Number of Loans", 0, 5, 1)

# Convert input to model format
input_data = pd.DataFrame({
    "Age": [age],
    "Occupation": [occupation],
    "Income": [income],
    "Banking_History": [banking_history],
    "Loans": [loans]
})

for col, encoder in encoders.items():
    input_data[col] = encoder.transform(input_data[col])
input_data = scaler.transform(input_data)

# Prediction
if st.sidebar.button("Predict Credit Score"):
    prediction = model.predict(input_data)[0]
    st.write(f"### Predicted Credit Score: {prediction}")
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feature_names = ["Age", "Occupation", "Income", "Banking_History", "Loans"]
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], ax=ax)
    st.pyplot(fig)
