# main.py

# Import functions from your other scripts
from data_preprocessing import preprocess_data
from train_model import train_linear_regression
from evaluate_model import evaluate_model

def main():
    # Step 1: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data('../data/boston_housing.csv')

    # Step 2: Train your model
    model = train_linear_regression(X_train, y_train)

    # Step 3: Evaluate your model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
