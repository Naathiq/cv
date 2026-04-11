# MULTIPLE LINEAR REGRESSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
data = pd.read_csv("dataset.txt", sep="\s+", header=None)

# Split into input (X) and output (y)
X = data.iloc[:, :-1].values   # all columns except last
y = data.iloc[:, -1].values    # last column

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Print coefficients
print("Coefficients:", model.coef_)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("Variance score (R²):", metrics.r2_score(y_test, y_pred))

# -----------------------------
# Plot residual errors
# -----------------------------

plt.style.use('fivethirtyeight')

# Training error
plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10, label="Train")

# Testing error
plt.scatter(model.predict(X_test),
            model.predict(X_test) - y_test,
            color="blue", s=10, label="Test")

# Zero error line
plt.plot(model.predict(X_train),
         np.zeros_like(model.predict(X_train)),
         color="red", linewidth=2)

plt.legend(loc='upper right')
plt.title("Residual Errors")
plt.show()