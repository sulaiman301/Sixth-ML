import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Function to load data
@st.cache
def load_data(file):
    df = pd.read_excel(file)
    return df

# Function to preprocess data
def preprocess_data(df):
    df["Churn Reason"] = df["Churn Reason"].fillna(df["Churn Reason"].mode()[0])
    df = df.drop(columns=['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Churn Reason'])
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)
    categorical_columns = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 
                           'Phone Service', 'Multiple Lines', 'Internet Service', 
                           'Online Security', 'Online Backup', 'Device Protection', 
                           'Tech Support', 'Streaming TV', 'Streaming Movies', 
                           'Contract', 'Paperless Billing', 'Payment Method']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df_encoded['Churn Label'] = df_encoded['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df_encoded

# Function to split data
def split_data(df):
    X = df.drop('Churn Label', axis=1)
    y = df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Function to train models
def train_models(X_train, y_train):
    log_reg = LogisticRegression()
    rf_clf = RandomForestClassifier()
    svc_clf = SVC()
    knn_clf = KNeighborsClassifier()
    models = {
        "Logistic Regression": log_reg,
        "Random Forest": rf_clf,
        "Support Vector Classifier": svc_clf,
        "K-Nearest Neighbors": knn_clf
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.write(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Function to plot confusion matrices
def plot_confusion_matrices(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

# Function to perform cross-validation
def cross_validate_models(models, X_train, y_train):
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        st.write(f"{name} - Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Function to print classification reports
def print_classification_reports(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        st.write(f"\n{name} Model Evaluation:")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

st.title("Customer Churn Analysis and Prediction")

# File upload
uploaded_file = "Data\Raw data\Telco_customer_churn.xlsx"
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data Loaded Successfully")
    
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # Data Preprocessing
    df_encoded = preprocess_data(df)
    st.write("Data Preprocessed Successfully")

    if st.checkbox("Show Preprocessed Data"):
        st.write(df_encoded.head())

    # Splitting and Scaling Data
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    X_train, X_test = scale_features(X_train, X_test)
    
    # Visualizations
    st.subheader("Data Visualization")
    
    numerical_features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']
    categorical_features = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 
                            'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 
                            'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 
                            'Contract', 'Paperless Billing', 'Payment Method', 'Churn Label']
    
    if st.checkbox("Show Univariate Analysis"):
        for feature in numerical_features:
            st.write(f"{feature}")
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            st.pyplot(fig)
        
        for feature in categorical_features:
            st.write(f"Count of {feature}")
            fig, ax = plt.subplots()
            sns.countplot(x=feature, data=df, ax=ax)
            st.pyplot(fig)
    
    if st.checkbox("Show Bivariate Analysis"):
        for feature in numerical_features:
            st.write(f"{feature} vs Churn Label")
            fig, ax = plt.subplots()
            sns.boxplot(x='Churn Label', y=feature, data=df, ax=ax)
            st.pyplot(fig)
        
        for feature in categorical_features:
            if feature != 'Churn Label':
                st.write(f"{feature} vs Churn Label")
                fig, ax = plt.subplots()
                sns.countplot(x=feature, hue='Churn Label', data=df, ax=ax)
                st.pyplot(fig)

    # Model Training and Evaluation
    st.subheader("Model Training and Evaluation")
    
    if st.button("Train and Evaluate Models"):
        models = train_models(X_train, y_train)
        st.write("Models Trained Successfully")
        
        st.write("Evaluating Models...")
        evaluate_models(models, X_test, y_test)
        
        st.write("Plotting Confusion Matrices...")
        plot_confusion_matrices(models, X_test, y_test)
        
        st.write("Performing Cross-Validation...")
        cross_validate_models(models, X_train, y_train)
        
        st.write("Classification Reports...")
        print_classification_reports(models, X_test, y_test)
