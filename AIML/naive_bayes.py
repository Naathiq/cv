from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create model
gnb = GaussianNB()

# Train model
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

# Accuracy
print("Gaussian Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)