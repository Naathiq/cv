# POLYNOMIAL LINEAR REGRESSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv("Position_Salaries.csv")

# Input (Position level) and Output (Salary)
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# -----------------------------
# SIMPLE LINEAR REGRESSION
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# -----------------------------
# POLYNOMIAL REGRESSION (Degree 2)
# -----------------------------
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2, y)

# -----------------------------
# POLYNOMIAL REGRESSION (Degree 3)
# -----------------------------
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)

lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3, y)

# -----------------------------
# PLOT SIMPLE LINEAR
# -----------------------------
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Simple Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# -----------------------------
# PLOT POLYNOMIAL (Degree 2 & 3)
# -----------------------------
plt.scatter(X, y, color='red')

# Smooth curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(-1,1)

plt.plot(X_grid, lin_reg2.predict(poly2.transform(X_grid)), color='green', label='Degree 2')
plt.plot(X_grid, lin_reg3.predict(poly3.transform(X_grid)), color='blue', label='Degree 3')

plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.show()