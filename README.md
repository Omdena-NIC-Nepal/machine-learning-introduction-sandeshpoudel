#Predicting House Prices Using Linear Regression

#Overview

This project demonstrates supervised machine learning through Linear Regression. It predicts house prices using various features from a real-world dataset.

Dataset
Source: Boston Housing Dataset (Custom dataset variant)
Data file: data/boston_housing.csv

Features:
price (Target variable)
Numeric features: area, bedrooms, bathrooms, stories, parking
Categorical features: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus

Project Structure
├── data
│   └── boston_housing.csv
├── notebooks
│   ├── EDA.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── Model_Training.ipynb
│   └── Model_Evaluation.ipynb
├── scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── main.py                  
├── README.md
├── requirements.txt
└── .gitignore


Setup and Installation

Clone the repository:

git clone https://github.com/Omdena-NIC-Nepal/machine-learning-introduction-sandeshpoudel.git


Create and activate the virtual environment:
python -m venv venv
# On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Workflow Steps

1. Exploratory Data Analysis (EDA)

Explore dataset
Visualize feature distributions
Check for outliers and missing values

2. Data Preprocessing
Handle outliers
Encode categorical variables
Standardize numeric features
Split dataset into training and testing

3. Model Training
Train Linear Regression model

4. Model Evaluation
Predict house prices on test data
Evaluate using metrics: Mean Squared Error (MSE), R-squared (R²)
Visualize actual vs predicted prices and residuals
Usage

You can reuse the scripts for quick processing:

from scripts.data_preprocessing import preprocess_data
from scripts.train_model import train_linear_regression
from scripts.evaluate_model import evaluate_model

X_train, X_test, y_train, y_test = preprocess_data('data/boston_housing.csv')
model = train_linear_regression(X_train, y_train)
evaluate_model(model, X_test, y_test)

Author
Sandesh Poudel
9855081056
contactsandesh.poudel@gmail.com

License
This project is open-source and available under the MIT License.