import numpy as np
import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load dataset (change path if needed)
heartDisease = pd.read_csv("heart.csv")

# Replace missing values
heartDisease = heartDisease.replace('?', np.nan)

# Display dataset
print("Few examples from dataset:")
print(heartDisease.head())

# Define Bayesian Network structure
model = BayesianNetwork([
    ('age', 'trestbps'),
    ('age', 'chol'),
    ('gender', 'trestbps'),
    ('trestbps', 'heartdisease'),
    ('chol', 'heartdisease'),
    ('fbs', 'heartdisease')
])

# Train model using Maximum Likelihood
print("\nLearning CPD using Maximum Likelihood Estimator")
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inference
print("\nInference with Bayesian Network")
infer = VariableElimination(model)

# Query 1: Probability of HeartDisease given Age
print("\n1. Probability of HeartDisease given age=63")
q1 = infer.query(variables=['heartdisease'], evidence={'age': 63})
print(q1)

# Query 2: Probability of HeartDisease given Cholesterol
print("\n2. Probability of HeartDisease given chol=204")
q2 = infer.query(variables=['heartdisease'], evidence={'chol': 204})
print(q2)