import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('stock_prices.csv')

# Split the data into features and target variable
X = data.drop('Close', axis=1)  # Features
y = data['Close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# Predict future stock prices
future_data = pd.read_csv('future_stock_prices.csv')
future_predictions = model.predict(future_data)

print("Future Stock Price Predictions:")
print(future_predictions)
