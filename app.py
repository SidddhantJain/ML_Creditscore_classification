import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import time

st.set_page_config(
    page_title="Credit Score Prediction App",
    page_icon="💳",
    layout="wide"
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .result-good {
        font-size: 2rem;
        color: #047857;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1FAE5;
        text-align: center;
    }
    .result-standard {
        font-size: 2rem;
        color: #9D174D;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FCE7F3;
        text-align: center;
    }
    .result-poor {
        font-size: 2rem;
        color: #991B1B;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FEE2E2;
        text-align: center;
    }
    .feature-section {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Credit Score Prediction System</h1>", unsafe_allow_html=True)

# Function to train model
@st.cache_resource
def train_model():
    # Check if model already exists
    if os.path.exists('random_forest_model.pkl'):
        return joblib.load('random_forest_model.pkl')
    
    # If in Colab, mount drive to access dataset
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        train_path = '/content/drive/MyDrive/Dataset (1)/train.csv'
    except:
        # If not in Colab, look for local file
        train_path = 'train.csv'
        
        # If file doesn't exist locally, use sample data
        if not os.path.exists(train_path):
            st.warning("Training dataset not found. Loading a sample dataset for demonstration.")
            # Create synthetic data for demonstration
            train_df = create_synthetic_data()
        else:
            train_df = pd.read_csv(train_path, dtype={26: str})
    
    # If path exists, load data
    if 'train_df' not in locals():
        train_df = pd.read_csv(train_path, dtype={26: str})
    
    # Data preprocessing
    drop_columns = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
    train_df.drop(columns=drop_columns, inplace=True, errors='ignore')
    
    # Handle Missing Values
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    train_df[numeric_cols] = train_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    train_df.fillna(train_df.median(numeric_only=True), inplace=True)
    
    # Encode Categorical Variables
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'Credit_Score' in categorical_cols:
        categorical_cols.remove('Credit_Score')
    
    # Create and fit encoders
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()
    
    if categorical_cols:
        train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols])
    
    if 'Credit_Score' in numeric_cols:
        numeric_cols.remove('Credit_Score')
    
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    
    # Save encoders for prediction
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Clean labels
    def clean_labels(label):
        if label not in ["Good", "Poor", "Standard"]:
            return "Standard"
        return label
    
    train_df['Credit_Score'] = train_df['Credit_Score'].astype(str).apply(clean_labels)
    
    # Split Data
    X = train_df.drop(columns=['Credit_Score'])
    y = train_df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance
    X_train.fillna(X_train.median(numeric_only=True), inplace=True)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(rf_model, 'random_forest_model.pkl')
    
    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')
    
    return rf_model

# Function to create synthetic data
def create_synthetic_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(20, 75, n_samples),
        'Annual_Income': np.random.randint(10000, 200000, n_samples),
        'Monthly_Inhand_Salary': np.random.randint(1000, 15000, n_samples),
        'Num_Bank_Accounts': np.random.randint(1, 10, n_samples),
        'Num_Credit_Card': np.random.randint(0, 7, n_samples),
        'Interest_Rate': np.random.randint(1, 35, n_samples),
        'Num_of_Loan': np.random.randint(0, 10, n_samples),
        'Delay_from_due_date': np.random.randint(0, 60, n_samples),
        'Num_of_Delayed_Payment': np.random.randint(0, 30, n_samples),
        'Changed_Credit_Limit': np.random.randint(0, 3, n_samples),
        'Num_Credit_Inquiries': np.random.randint(0, 10, n_samples),
        'Outstanding_Debt': np.random.randint(0, 50000, n_samples),
        'Credit_Utilization_Ratio': np.random.uniform(0, 1, n_samples),
        'Credit_History_Age': np.random.randint(1, 400, n_samples),
        'Payment_of_Min_Amount': np.random.choice(['Yes', 'No'], n_samples),
        'Total_EMI_per_month': np.random.randint(0, 5000, n_samples),
        'Amount_invested_monthly': np.random.randint(0, 10000, n_samples),
        'Monthly_Balance': np.random.randint(-1000, 50000, n_samples),
        'Occupation': np.random.choice(['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer'], n_samples),
        'Type_of_Loan': np.random.choice(['Auto Loan', 'Credit-Builder Loan', 'Personal Loan', 'Home Equity Loan', 'Mortgage Loan', 'Student Loan', 'Debt Consolidation Loan', 'Payday Loan'], n_samples),
        'Credit_Mix': np.random.choice(['Good', 'Standard', 'Bad'], n_samples),
        'Payment_Behaviour': np.random.choice(['High_spent_Small_value_payments', 'Low_spent_Large_value_payments', 'High_spent_Large_value_payments', 'Low_spent_Small_value_payments'], n_samples),
        'Credit_Score': np.random.choice(['Good', 'Poor', 'Standard'], n_samples)
    }
    
    return pd.DataFrame(data)

# Load or train model
with st.spinner('Loading model...'):
    model = train_model()
    
    # Load encoders and feature names if they exist
    try:
        encoder = joblib.load('encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
    except:
        st.error("Error loading encoders or feature names. Please run the training code first.")
        feature_names = []

# Function to preprocess input data
def preprocess_input(data):
    # Convert data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Identify categorical and numerical columns
    categorical_cols = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
    numeric_cols = [col for col in input_df.columns if col not in categorical_cols]
    
    # Handle categorical variables
    try:
        input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    except:
        # If encoder fails, use simple encoding
        for col in categorical_cols:
            if col == 'Payment_of_Min_Amount':
                input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})
            elif col == 'Credit_Mix':
                input_df[col] = input_df[col].map({'Good': 2, 'Standard': 1, 'Bad': 0})
            else:
                # For other categorical columns, use position in the list of unique values
                unique_vals = input_df[col].unique()
                input_df[col] = input_df[col].map({val: i for i, val in enumerate(unique_vals)})
    
    # Scale numerical variables
    try:
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    except:
        # If scaler fails, use simple standardization
        for col in numeric_cols:
            mean = input_df[col].mean()
            std = input_df[col].std() if input_df[col].std() != 0 else 1
            input_df[col] = (input_df[col] - mean) / std
    
    # Make sure the column order matches the model's expected features
    if feature_names:
        input_df = input_df[feature_names]
    
    return input_df

# Function to predict credit score
def predict_credit_score(input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    
    # Get prediction probability
    proba = model.predict_proba(processed_data)[0]
    proba_dict = {
        'Poor': round(proba[model.classes_.tolist().index('Poor')] * 100, 2) if 'Poor' in model.classes_ else 0,
        'Standard': round(proba[model.classes_.tolist().index('Standard')] * 100, 2) if 'Standard' in model.classes_ else 0,
        'Good': round(proba[model.classes_.tolist().index('Good')] * 100, 2) if 'Good' in model.classes_ else 0
    }
    
    return prediction, proba_dict

# Create sidebar for inputs
st.sidebar.markdown("<h2 class='sub-header'>Enter Your Information</h2>", unsafe_allow_html=True)

# Personal Information
st.sidebar.markdown("<div class='feature-section'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Personal Information</h3>", unsafe_allow_html=True)
name = st.sidebar.text_input("Name (not used in prediction)", "John Doe")
age = st.sidebar.slider("Age", 18, 100, 30)
occupation = st.sidebar.selectbox(
    "Occupation", 
    ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer", "Doctor", "Accountant", "Banker", "Other"]
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Financial Information
st.sidebar.markdown("<div class='feature-section'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Income & Savings</h3>", unsafe_allow_html=True)
annual_income = st.sidebar.number_input("Annual Income ($)", 0, 1000000, 50000, step=1000)
monthly_salary = st.sidebar.number_input("Monthly Inhand Salary ($)", 0, 100000, 4000, step=100)
monthly_balance = st.sidebar.number_input("Monthly Balance ($)", -10000, 100000, 1000, step=100)
amount_invested = st.sidebar.number_input("Amount Invested Monthly ($)", 0, 50000, 500, step=100)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Banking Information
st.sidebar.markdown("<div class='feature-section'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Banking Details</h3>", unsafe_allow_html=True)
num_bank_accounts = st.sidebar.slider("Number of Bank Accounts", 0, 20, 2)
num_credit_cards = st.sidebar.slider("Number of Credit Cards", 0, 10, 1)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0, 40, 10)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Loan Information
st.sidebar.markdown("<div class='feature-section'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Loan Information</h3>", unsafe_allow_html=True)
num_loans = st.sidebar.slider("Number of Loans", 0, 10, 1)
loan_type = st.sidebar.selectbox(
    "Type of Loan", 
    ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan", 
     "Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan", "None"]
)
total_emi = st.sidebar.number_input("Total EMI per Month ($)", 0, 10000, 500, step=100)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Credit Information
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
    st.markdown("<h3>Credit History</h3>", unsafe_allow_html=True)
    credit_history_age = st.number_input("Credit History Age (months)", 0, 600, 100, step=10)
    credit_utilization = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3, step=0.01)
    credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
    changed_credit_limit = st.slider("Changed Credit Limit", 0, 3, 1)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
    st.markdown("<h3>Payment Behavior</h3>", unsafe_allow_html=True)
    num_delayed_payments = st.slider("Number of Delayed Payments", 0, 50, 5)
    delay_from_due_date = st.slider("Delay from Due Date (days)", 0, 100, 10)
    payment_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"])
    payment_behavior = st.selectbox(
        "Payment Behavior", 
        ["High_spent_Small_value_payments", "Low_spent_Large_value_payments", 
         "High_spent_Large_value_payments", "Low_spent_Small_value_payments"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Additional Credit Information
st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
st.markdown("<h3>Additional Credit Information</h3>", unsafe_allow_html=True)
num_credit_inquiries = st.slider("Number of Credit Inquiries", 0, 20, 2)
outstanding_debt = st.number_input("Outstanding Debt ($)", 0, 1000000, 10000, step=1000)
ssn = st.text_input("SSN (not used in prediction)", "123-45-6789")
st.markdown("</div>", unsafe_allow_html=True)

# Create input data dictionary
input_data = {
    'Age': age,
    'Annual_Income': annual_income,
    'Monthly_Inhand_Salary': monthly_salary,
    'Num_Bank_Accounts': num_bank_accounts,
    'Num_Credit_Card': num_credit_cards,
    'Interest_Rate': interest_rate,
    'Num_of_Loan': num_loans,
    'Type_of_Loan': loan_type,
    'Delay_from_due_date': delay_from_due_date,
    'Num_of_Delayed_Payment': num_delayed_payments,
    'Changed_Credit_Limit': changed_credit_limit,
    'Num_Credit_Inquiries': num_credit_inquiries,
    'Credit_Mix': credit_mix,
    'Outstanding_Debt': outstanding_debt,
    'Credit_Utilization_Ratio': credit_utilization,
    'Credit_History_Age': credit_history_age,
    'Payment_of_Min_Amount': payment_min_amount,
    'Total_EMI_per_month': total_emi,
    'Amount_invested_monthly': amount_invested,
    'Payment_Behaviour': payment_behavior,
    'Monthly_Balance': monthly_balance,
    'Occupation': occupation
}

# Predict button
if st.button('Predict Credit Score', key='predict_button'):
    # Show spinner during prediction
    with st.spinner('Calculating your credit score...'):
        time.sleep(1)  # Simulate computation time
        prediction, probabilities = predict_credit_score(input_data)
    
    # Display result
    st.markdown("## Prediction Result")
    
    if prediction == 'Good':
        st.markdown(f"<div class='result-good'>Your predicted credit score is: {prediction}</div>", unsafe_allow_html=True)
    elif prediction == 'Standard':
        st.markdown(f"<div class='result-standard'>Your predicted credit score is: {prediction}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-poor'>Your predicted credit score is: {prediction}</div>", unsafe_allow_html=True)
    
    # Display probabilities
    st.markdown("### Prediction Confidence")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Poor", f"{probabilities['Poor']}%")
    
    with col2:
        st.metric("Standard", f"{probabilities['Standard']}%")
    
    with col3:
        st.metric("Good", f"{probabilities['Good']}%")
    
    # Display feature importance chart
    st.markdown("### Feature Importance")
    st.info("This chart shows which factors most influenced your credit score prediction.")
    
    # Extract feature importances
    feature_importances = model.feature_importances_
    
    # Get top 10 features
    feature_names = list(input_data.keys())
    top_indices = np.argsort(feature_importances)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = [feature_importances[i] for i in top_indices]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features, top_importances, color='#3B82F6')
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Features Influencing Credit Score')
    st.pyplot(fig)
    
    # Add recommendations based on prediction
    st.markdown("### Recommendations")
    
    if prediction == 'Good':
        st.success("""
        Congratulations on your excellent credit score! Here are some tips to maintain it:
        - Continue making payments on time
        - Maintain low credit utilization
        - Be selective about opening new credit accounts
        - Monitor your credit report regularly
        """)
    elif prediction == 'Standard':
        st.info("""
        Your credit score is acceptable, but there's room for improvement:
        - Pay down existing debt
        - Make all payments on time
        - Avoid applying for multiple new credit lines
        - Reduce credit utilization to below 30%
        - Build a more diverse credit mix
        """)
    else:
        st.warning("""
        Here are some steps to improve your credit score:
        - Set up automatic payments to avoid missed deadlines
        - Create a debt repayment plan focusing on high-interest accounts
        - Avoid applying for new credit
        - Consider a secured credit card to rebuild credit
        - Check your credit report for errors
        - Be patient - improvement takes time
        """)

else:
    # Display information before prediction
    st.markdown("""
    ## How It Works
    
    This application uses machine learning to predict your credit score based on various financial and personal factors.
    
    1. Fill in your information in the sidebar
    2. Click the 'Predict Credit Score' button
    3. Get your predicted credit rating (Poor, Standard, or Good)
    4. Review recommendations to improve or maintain your score
    
    ### Why Predict Your Credit Score?
    
    Understanding your credit score can help you:
    - Plan for future loan applications
    - Identify areas for financial improvement
    - Negotiate better interest rates
    - Monitor your financial health
    
    **Note:** This tool provides an estimation only. Actual credit scores may vary based on additional factors used by credit bureaus.
    """)

# Add footer
st.markdown("""
---
### About This Application

This credit score prediction system uses a Random Forest Classifier trained on financial data to predict credit scores.
The model classifies credit scores into three categories: Poor, Standard, and Good.
""")

# Add instructions for Colab/Codespace
with st.expander("Installation Instructions"):
    st.markdown("""
    ### Running this app in Google Colab:
    
    ```python
    !pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib imbalanced-learn
    !pip install streamlit-option-menu
    
    # Save this script as app.py
    # Then run:
    !streamlit run app.py & npx localtunnel --port 8501
    ```
    
    ### Running in GitHub Codespace:
    
    ```
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib imbalanced-learn
    streamlit run app.py
    ```
    """)
