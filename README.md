## Telco Customer Churn Prediction
# Overview
This project aims to predict customer churn for a telecommunications company. The dataset contains various customer attributes, including demographics, services subscribed, and account information. The goal is to build a machine learning model that can accurately predict whether a customer will churn (i.e., leave the service) based on these attributes.

# Project Structure

bash
Copy code
├── data/
│   ├── Raw data/
│   │   └── Telco_customer_churn.xlsx  # Raw dataset used for the project
├── notebooks/
│   └── churn_prediction.ipynb          # Jupyter notebook with analysis and model development
├── src/
│   └── model.py                        # Python script for model training and evaluation
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies


# Data
The dataset used in this project is Telco_customer_churn.xlsx, which contains the following key features:

CustomerID: Unique identifier for each customer.
Tenure Months: Number of months the customer has been with the company.
Monthly Charges: The amount charged to the customer monthly.
Total Charges: Total amount charged during the tenure.
Churn Label: Target variable indicating whether the customer churned (Yes) or not (No).
Various Categorical Features: Such as Gender, Senior Citizen, Partner, Dependents, Internet Service, etc.
Preprocessing
The data preprocessing steps include:

Missing Value Handling: Imputed missing values in the Churn Reason column with the mode.
Dropping Irrelevant Columns: Removed columns like CustomerID, Count, Country, etc., which are not relevant to the model.
Encoding Categorical Variables: Used one-hot encoding to convert categorical variables into numerical format.
Feature Scaling: Applied standard scaling to normalize the numerical features.
Exploratory Data Analysis (EDA)
The following analyses were performed:

# Univariate Analysis:
Examined the distribution of individual variables using histograms and count plots.

# Bivariate Analysis:
Analyzed relationships between features and the target variable using box plots and count plots.

# Multivariate Analysis:
Conducted correlation analysis to understand relationships among features.

# Machine Learning Models
Four machine learning models were trained and evaluated:

Logistic Regression
Random Forest
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)

# Model Evaluation
The models were evaluated using the following metrics:

Accuracy
Precision
Recall
F1-Score

# Confusion Matrix
Cross-validation was performed to assess the stability and generalizability of the models.

# Results
The evaluation metrics for each model were compared, with the best-performing model selected based on accuracy, precision, recall, and F1 score.
Confusion matrices were plotted to visualize the performance of each model in distinguishing between churned and non-churned customers.
Future Work

# Hyperparameter Tuning:
Further optimization of model performance through techniques like Grid Search.

# Feature Importance Analysis: 

Understanding which features are most important in predicting churn.
# Handling Class Imbalance: 

Implementing techniques to handle imbalanced data, such as SMOTE or adjusting class weights.
# Dependencies
To install the necessary dependencies, run:

bash
Copy code
pip install -r requirements.txt

# License
This project is licensed under the MIT License.

# Acknowledgements
This project was inspired by the need to improve customer retention strategies by understanding the key factors leading to churn.

