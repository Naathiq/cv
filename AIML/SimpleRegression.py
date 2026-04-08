# SIMPLE LINEAR REGRESSION

import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    # number of observations
    n = np.size(x)

    # mean of x and y
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # regression coefficients
    b1 = SS_xy / SS_xx
    b0 = m_y - b1 * m_x

    return (b0, b1)

def plot_regression_line(x, y, b):
    # scatter plot
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predicted values
    y_pred = b[0] + b[1] * x

    # regression line
    plt.plot(x, y_pred, color="g")

    # labels
    plt.xlabel('X')
    plt.ylabel('Y')

def main():
    # data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimate coefficients
    b = estimate_coef(x, y)

    print("Estimated coefficients:")
    print("b0 =", b[0])
    print("b1 =", b[1])

    # plot
    plot_regression_line(x, y, b)
    plt.show()

if __name__ == "__main__":
    main()