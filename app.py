import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('house-prices.csv')

# Drop unnecessary columns
data = data.drop(['date', 'area', 'code', 'borough_flag'], axis=1)

# Impute missing values with the mean of the column
data = data.fillna(data.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('average_price', axis=1), data['average_price'], test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

# Visualize the results
plt.bar(np.arange(len(y_test)), y_test, color='b', label='Actual')
plt.bar(np.arange(len(y_pred)), y_pred, color='g', label='Predicted')
plt.xlabel('House index')
plt.ylabel('House price')
plt.legend()
plt.show()
