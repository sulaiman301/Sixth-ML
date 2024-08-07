pip install streamlit
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from functions import load_data, preprocess_data, split_data, scale_features, plot_univariate, plot_bivariate, train_models, evaluate_models, plot_confusion_matrices, cross_validate_models, print_classification_reports

st.title("Customer Churn Analysis and Prediction")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
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
            st.write(f"Distribution of {feature}")
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

if __name__ == '__main__':
    st.run()

streamlit run app.py
