# evaluate_model.py
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics clearly
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared Score: {r2:.2f}')

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()
