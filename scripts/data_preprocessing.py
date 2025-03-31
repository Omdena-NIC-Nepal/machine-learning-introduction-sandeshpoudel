# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Handling outliers
    upper_limit = df['price'].quantile(0.99)
    df['price'] = np.where(df['price'] > upper_limit, upper_limit, df['price'])

    # Encoding binary categorical variables
    binary_vars = ['mainroad', 'guestroom', 'basement', 
                   'hotwaterheating', 'airconditioning', 'prefarea']
    for var in binary_vars:
        df[var] = df[var].map({'yes':1, 'no':0})

    # Encoding 'furnishingstatus' (multiple categories)
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Split dataset clearly
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
